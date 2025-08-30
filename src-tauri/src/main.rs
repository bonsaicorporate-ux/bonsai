#![allow(clippy::needless_return)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tauri::Manager;
use tauri::path::BaseDirectory;

use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{
    Config as QwenCfg,
    ModelForCausalLM as QwenLM,  // <-- use the LM head variant
  };
use memmap2::Mmap;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use candle_transformers::generation::LogitsProcessor;
// use rand::thread_rng;

#[derive(Clone)]
struct Paths {
  model_dir: PathBuf,
  weights: PathBuf,
  tokenizer_json: PathBuf,
  config_json: PathBuf,
}

// 1) extend Llm
struct Llm {
    dev: Device,
    tok: Tokenizer,
    cfg: QwenCfg,
    model: QwenLM,
    weights_path: PathBuf,   // <-- add this
  }

struct AppState {
    llm: Arc<Mutex<Llm>>,
  }

#[tauri::command]
fn echo(text: String) -> String {
  text
}

fn resolve_model_paths(app: &tauri::AppHandle) -> Result<Paths, String> {
  let model_dir = app
    .path()
    .resolve("deepseek-model", BaseDirectory::Resource)
    .map_err(|e| format!("resolve model dir: {e}"))?;

  let p = |name: &str| model_dir.join(name);

  let paths = Paths {
    model_dir: model_dir.clone(),
    weights: p("model.safetensors"),
    tokenizer_json: p("tokenizer.json"),
    config_json: p("config.json"),
  };

  for (label, path) in [
    ("weights", &paths.weights),
    ("tokenizer.json", &paths.tokenizer_json),
    ("config.json", &paths.config_json),
  ] {
    if !path.exists() {
      return Err(format!("Missing {label} at {}", path.display()));
    }
  }

  Ok(paths)
}

fn load_llm(app: &tauri::AppHandle) -> Result<Llm, String> {
    // Prefer Apple GPU via Metal; fallback to CPU
    let dev = candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu);
  
    // Resolve model files
    let paths = resolve_model_paths(app)?;
  
    // Tokenizer
    let tok = Tokenizer::from_file(paths.tokenizer_json.to_string_lossy().to_string())
      .map_err(|e| format!("load tokenizer: {e}"))?;
  
    // Config
    let cfg_bytes = std::fs::read(&paths.config_json)
      .map_err(|e| format!("read config.json: {e}"))?;
    let cfg: QwenCfg = serde_json::from_slice(&cfg_bytes)
      .map_err(|e| format!("parse config.json: {e}"))?;
  
    // (Optional) quick safetensors header check
    {
      use std::fs::File;
      let f = File::open(&paths.weights).map_err(|e| format!("open weights: {e}"))?;
      let mmap = unsafe { Mmap::map(&f).map_err(|e| format!("mmap weights: {e}"))? };
      let _ = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;
    }
  
    let weights_path = paths.weights.clone();
  
    // Use F16 on Metal, but F32 on CPU for numerical stability
    let target_dtype = if matches!(dev, candle_core::Device::Cpu) { DType::F32 } else { DType::F16 };
  
    let vb = unsafe {
      VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], target_dtype, &dev)
        .map_err(|e| format!("varbuilder from safetensors: {e}"))?
    };
  
    let model = QwenLM::new(&cfg, vb).map_err(|e| format!("construct Qwen2 LM: {e}"))?;
  
    Ok(Llm { dev, tok, cfg, model, weights_path })
  }

  #[tauri::command]
fn init_model(app: tauri::AppHandle) -> Result<String, String> {
  let llm = load_llm(&app)?;
  app.manage(AppState { llm: Arc::new(Mutex::new(llm)) });
  Ok("ready".into())
}

fn rebuild_model(llm: &mut Llm) -> Result<(), String> {
    let target_dtype = if matches!(llm.dev, candle_core::Device::Cpu) { DType::F32 } else { DType::F16 };
    let vb = unsafe {
      VarBuilder::from_mmaped_safetensors(&[llm.weights_path.clone()], target_dtype, &llm.dev)
        .map_err(|e| format!("varbuilder from safetensors: {e}"))?
    };
    llm.model = QwenLM::new(&llm.cfg, vb)
      .map_err(|e| format!("reconstruct Qwen2 LM: {e}"))?;
    Ok(())
  }

  #[tauri::command]
  fn generate(app: tauri::AppHandle, prompt: String) -> Result<(), String> {
    use tauri::Emitter;
    use candle_core::IndexOp;
  
    // ---- get state & reset KV cache for a clean generation ----
    let state: tauri::State<'_, AppState> = app.state();
    let mut llm = state.llm.lock().map_err(|_| "model mutex poisoned".to_string())?;
    rebuild_model(&mut llm)?; // clears cache by reconstructing the LM head
  
    // ---- locals ----
    let dev = llm.dev.clone();       // keep Device::Cpu in load_llm() while debugging
    let mut tok = llm.tok.clone();
    let model = &mut llm.model;
  
    let _ = app.emit("token", "▶︎ start ");
  
    // ---- tokenize prompt ----
    let enc = tok.encode(prompt, true).map_err(|e| e.to_string())?;
    let mut ids_u32: Vec<u32> = enc.get_ids().to_vec();
    if ids_u32.is_empty() { ids_u32.push(1); }
    let mut ids_i64: Vec<i64> = ids_u32.iter().copied().map(|x| x as i64).collect();
  
    // ---- sampler (temperature + top-p) ----
    let mut sampler = candle_transformers::generation::LogitsProcessor::new(
      /*seed*/ 42,
      Some(0.9),   // temperature
      Some(0.95),  // top_p
    );
  
    let max_new_tokens: usize = 32;
  
    // ========= 1) First pass with full prompt (offset = 0) =========
    let prompt_len = ids_i64.len();
    let xs0 = candle_core::Tensor::new(ids_i64.as_slice(), &dev).map_err(|e| e.to_string())?;
    let xs  = xs0.reshape((1, prompt_len)).map_err(|e| e.to_string())?;
  
    // logits: [1, prompt_len, vocab]
    let logits = match model.forward(&xs, 0) {
      Ok(l) => l,
      Err(e) => { let _ = app.emit("token", format!("\n[ERR] forward(prompt): {e}")); return Ok(()); }
    };
  
    // take last step → [vocab], cast to f32 for stable sampling
    let last = match logits.i((0usize, prompt_len - 1)) {
      Ok(t) => t,
      Err(e) => { let _ = app.emit("token", format!("\n[ERR] index last(prompt): {e}")); return Ok(()); }
    };
    let last = match last.to_dtype(candle_core::DType::F32) {
      Ok(t) => t,
      Err(e) => { let _ = app.emit("token", format!("\n[ERR] cast(prompt): {e}")); return Ok(()); }
    };
  
    // sample first token
    let mut next_id: u32 = match sampler.sample(&last) {
      Ok(id) => id as u32,
      Err(e) => {
        // fallback to argmax if sampling complains
        let _ = app.emit("token", format!("\n[sample→argmax prompt] {e}"));
        let idx_t = match last.argmax(0) {
          Ok(t) => t,
          Err(e) => { let _ = app.emit("token", format!("\n[ERR] argmax(prompt): {e}")); return Ok(()); }
        };
        match idx_t.dtype() {
          candle_core::DType::U32 => idx_t.to_scalar::<u32>().map_err(|e| e.to_string())?,
          candle_core::DType::I64 => idx_t.to_scalar::<i64>().map_err(|e| e.to_string())? as u32,
          other => { let _ = app.emit("token", format!("\n[ERR] argmax dtype (prompt): {:?}", other)); return Ok(()); }
        }
      }
    };
  
    // stream piece
    if let Ok(piece) = tok.decode(&[next_id], false) {
      let _ = app.emit("token", if piece.is_empty() { format!("[id:{next_id}]") } else { piece });
    }
  
    // append & set seqlen_offset for cache
    ids_u32.push(next_id);
    ids_i64.push(next_id as i64);
    let mut seqlen_offset = prompt_len;
  
    // ========= 2) Incremental steps: 1 token per forward =========
    for _ in 1..max_new_tokens {
      // feed only the last token with current offset
      let step_ids = [*ids_i64.last().unwrap()];
      let xs0 = candle_core::Tensor::new(&step_ids, &dev).map_err(|e| e.to_string())?;
      let xs  = xs0.reshape((1, 1)).map_err(|e| e.to_string())?;
  
      // logits: [1, 1, vocab]
      let logits = match model.forward(&xs, seqlen_offset) {
        Ok(l) => l,
        Err(e) => { let _ = app.emit("token", format!("\n[ERR] forward(step @ {seqlen_offset}): {e}")); break; }
      };
  
      // index to [vocab], cast to f32
      let last = match logits.i((0usize, 0usize)) {
        Ok(t) => t,
        Err(e) => { let _ = app.emit("token", format!("\n[ERR] index last(step): {e}")); break; }
      };
      let last = match last.to_dtype(candle_core::DType::F32) {
        Ok(t) => t,
        Err(e) => { let _ = app.emit("token", format!("\n[ERR] cast(step): {e}")); break; }
      };
  
      // sample next token (fallback to argmax)
      next_id = match sampler.sample(&last) {
        Ok(id) => id as u32,
        Err(e) => {
          let _ = app.emit("token", format!("\n[sample→argmax step] {e}"));
          let idx_t = match last.argmax(0) {
            Ok(t) => t,
            Err(e) => { let _ = app.emit("token", format!("\n[ERR] argmax(step): {e}")); break; }
          };
          match idx_t.dtype() {
            candle_core::DType::U32 => match idx_t.to_scalar::<u32>() {
              Ok(v) => v,
              Err(e) => { let _ = app.emit("token", format!("\n[ERR] to_scalar<u32>(step): {e}")); break; }
            },
            candle_core::DType::I64 => match idx_t.to_scalar::<i64>() {
              Ok(v) => v as u32,
              Err(e) => { let _ = app.emit("token", format!("\n[ERR] to_scalar<i64>(step): {e}")); break; }
            },
            other => { let _ = app.emit("token", format!("\n[ERR] argmax dtype(step): {:?}", other)); break; }
          }
        }
      };
  
      if let Ok(piece) = tok.decode(&[next_id], false) {
        let _ = app.emit("token", if piece.is_empty() { format!("[id:{next_id}]") } else { piece });
      }
  
      ids_u32.push(next_id);
      ids_i64.push(next_id as i64);
      seqlen_offset += 1;
    }
  
    let _ = app.emit("token", "\n✓ done");
    Ok(())
  }

fn main() {
  tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![echo, init_model, generate])
    .run(tauri::generate_context!())
    .expect("error while running tauri app");
}