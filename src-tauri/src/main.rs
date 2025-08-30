#![allow(clippy::needless_return)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tauri::Manager;
use tauri::path::BaseDirectory;

use candle_core::{Device, DType, Tensor, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{
  Config as QwenCfg,
  ModelForCausalLM as QwenLM,
};

use memmap2::Mmap;
use safetensors::SafeTensors;
use tokenizers::{Tokenizer, AddedToken};

// ---------- paths & app state ----------

#[derive(Clone)]
struct Paths {
  model_dir: PathBuf,
  weights: PathBuf,
  tokenizer_json: PathBuf,
  config_json: PathBuf,
}

struct Llm {
  dev: Device,
  tok: Tokenizer,
  cfg: QwenCfg,
  model: QwenLM,
  weights_path: PathBuf,
}

struct AppState {
  llm: Arc<Mutex<Llm>>,
}

#[tauri::command]
fn echo(text: String) -> String { text }

// ---------- helpers ----------

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
  // Prefer Metal, fallback to CPU
  let dev = Device::new_metal(0).unwrap_or(Device::Cpu);

  let paths = resolve_model_paths(app)?;

  // Tokenizer
  let tok = Tokenizer::from_file(paths.tokenizer_json.to_string_lossy().to_string())
    .map_err(|e| format!("load tokenizer: {e}"))?;

  // Config
  let cfg_bytes = std::fs::read(&paths.config_json)
    .map_err(|e| format!("read config.json: {e}"))?;
  let cfg: QwenCfg = serde_json::from_slice(&cfg_bytes)
    .map_err(|e| format!("parse config.json: {e}"))?;

  // Quick safetensors header check
  {
    use std::fs::File;
    let f = File::open(&paths.weights).map_err(|e| format!("open weights: {e}"))?;
    let mmap = unsafe { Mmap::map(&f).map_err(|e| format!("mmap weights: {e}"))? };
    let _ = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse safetensors: {e}"))?;
  }

  let weights_path = paths.weights.clone();

  // DType: F32 on CPU for numeric stability, F16 on Metal
  let target_dtype = if matches!(dev, Device::Cpu) { DType::F32 } else { DType::F16 };

  let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], target_dtype, &dev)
      .map_err(|e| format!("varbuilder from safetensors: {e}"))?
  };

  let model = QwenLM::new(&cfg, vb).map_err(|e| format!("construct Qwen2 LM: {e}"))?;

  Ok(Llm { dev, tok, cfg, model, weights_path })
}

fn rebuild_model(llm: &mut Llm) -> Result<(), String> {
  // Rebuild to clear KV cache, keeping device & dtype choice.
  let target_dtype = if matches!(llm.dev, Device::Cpu) { DType::F32 } else { DType::F16 };
  let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&[llm.weights_path.clone()], target_dtype, &llm.dev)
      .map_err(|e| format!("varbuilder from safetensors: {e}"))?
  };
  llm.model = QwenLM::new(&llm.cfg, vb)
    .map_err(|e| format!("reconstruct Qwen2 LM: {e}"))?;
  Ok(())
}

// Pick last step logits across common shapes
fn pick_last_step(logits: &Tensor) -> Result<Tensor, String> {
  let d = logits.dims().to_vec();
  match d.as_slice() {
    // [B,S,V] → take last S
    [b, s, _v] if *b == 1 && *s >= 1 => logits.i((0usize, s - 1)).map_err(|e| e.to_string()),
    // [B,V]
    [b, _v] if *b == 1 => logits.i(0usize).map_err(|e| e.to_string()),
    // [S,V]
    [s, _v] if *s >= 1 => logits.i(s - 1).map_err(|e| e.to_string()),
    // [V]
    [_v] => Ok(logits.clone()),
    other => Err(format!("unexpected logits dims: {other:?}")),
  }
}

// ---------- commands ----------

#[tauri::command]
fn init_model(app: tauri::AppHandle) -> Result<String, String> {
  let llm = load_llm(&app)?;
  app.manage(AppState { llm: Arc::new(Mutex::new(llm)) });
  Ok("ready".into())
}

#[tauri::command]
fn generate(app: tauri::AppHandle, prompt: String) -> Result<(), String> {
  // Encourage R1-style reasoning (DeepSeek’s README suggests starting with <think>)
  let prompt = format!("{}\n<think>\n", prompt.trim());

  let app_for_thread = app.clone();
  std::thread::spawn(move || {
    use tauri::Emitter;

    // timing
    let t0 = std::time::Instant::now();
    let mut n_tokens: usize = 0;

    // lock & reset KV cache (cheap & reliable)
    let state: tauri::State<'_, AppState> = app_for_thread.state();
    let mut llm = match state.llm.lock() {
      Ok(g) => g,
      Err(_) => { let _ = app_for_thread.emit("token", "[ERR] model lock"); return; }
    };
    if let Err(e) = rebuild_model(&mut llm) {
      let _ = app_for_thread.emit("token", format!("[ERR] rebuild: {e}"));
      return;
    }

    // locals
    let dev   = llm.dev.clone();
    let mut tok = llm.tok.clone(); // own tokenizer (we will pad it if needed)
    let model = &mut llm.model;

    let _ = app_for_thread.emit("token", "▶︎ start ");

    // tokenize (cap prefill during dev for speed)
    let enc = match tok.encode(prompt, true) {
      Ok(e) => e,
      Err(e) => { let _ = app_for_thread.emit("token", format!("[ERR] tokenize: {e}")); return; }
    };
    let mut ids_u32: Vec<u32> = enc.get_ids().to_vec();
    if ids_u32.is_empty() { ids_u32.push(1); }

    const MAX_PREFILL_TOKENS: usize = 128; // raise later
    if ids_u32.len() > MAX_PREFILL_TOKENS {
      ids_u32 = ids_u32[ids_u32.len() - MAX_PREFILL_TOKENS..].to_vec();
    }
    let mut ids_i64: Vec<i64> = ids_u32.iter().map(|&x| x as i64).collect();

    // sampler: start greedy for stability (you can switch to Some(0.6), Some(0.95))
    let mut sampler = LogitsProcessor::new(42, None, None);

    // === prefill (offset=0) ===
    let prompt_len = ids_i64.len();
    let xs0 = match Tensor::new(ids_i64.as_slice(), &dev) {
      Ok(t) => t, Err(e) => { let _ = app_for_thread.emit("token", format!("[ERR] tensor(prompt): {e}")); return; }
    };
    let xs  = match xs0.reshape((1, prompt_len)) {
      Ok(t) => t, Err(e) => { let _ = app_for_thread.emit("token", format!("[ERR] reshape(prompt): {e}")); return; }
    };
    let logits = match model.forward(&xs, 0) {
      Ok(l) => l, Err(e) => { let _ = app_for_thread.emit("token", format!("[ERR] forward(prompt): {e}")); return; }
    };

    // If tokenizer vocab < model vocab, pad tokenizer with dummy specials so decode never fails.
    let tok_vocab = tok.get_vocab(true).len();
    let model_vocab = *logits.dims().last().unwrap_or(&0);
    if tok_vocab < model_vocab {
      let missing = model_vocab - tok_vocab;
      let mut extras = Vec::with_capacity(missing);
      for i in 0..missing {
        extras.push(AddedToken::from(format!("<extra_{i}>"), true));
      }
      tok.add_special_tokens(&extras);
    }

    // pick last step & cast to f32 for stable sampling
    let mut last = match pick_last_step(&logits) {
      Ok(t) => t, Err(e) => { let _ = app_for_thread.emit("token", format!("[ERR] index(prompt): {e}")); return; }
    };
    last = match last.to_dtype(DType::F32) {
      Ok(t) => t, Err(e) => { let _ = app_for_thread.emit("token", format!("[ERR] cast(prompt): {e}")); return; }
    };
    if let Ok(t) = last.clamp(-50f32, 50f32) { last = t; }

    // sample first token (greedy), fallback to argmax
    let mut next_id: u32 = match sampler.sample(&last) {
      Ok(id) => id as u32,
      Err(_) => match last.argmax(0)
        .and_then(|t| t.to_scalar::<u32>().or_else(|_| t.to_scalar::<i64>().map(|v| v as u32))) {
        Ok(v) => v, Err(_) => 0
      }
    };

    // stream the first token (show raw id if empty decode)
    {
      let piece = tok.decode(&[next_id], false).unwrap_or_default();
      if piece.is_empty() {
        let _ = app_for_thread.emit("token", format!("[id:{next_id}]"));
      } else {
        let _ = app_for_thread.emit("token", piece);
      }
      n_tokens += 1;
    }

    // prepare for incremental decoding
    ids_u32.push(next_id);
    ids_i64.push(next_id as i64);
    let mut seqlen_offset = prompt_len;

    let max_new_tokens: usize = 24;

    // === decode loop (KV cache) ===
    for _ in 1..max_new_tokens {
      let step_ids = [*ids_i64.last().unwrap()];
      let xs0 = match Tensor::new(&step_ids, &dev) {
        Ok(t) => t, Err(e) => { let _ = app_for_thread.emit("token", format!("\n[ERR] tensor(step): {e}")); break; }
      };
      let xs  = match xs0.reshape((1, 1)) {
        Ok(t) => t, Err(e) => { let _ = app_for_thread.emit("token", format!("\n[ERR] reshape(step): {e}")); break; }
      };
      let logits = match model.forward(&xs, seqlen_offset) {
        Ok(l) => l, Err(e) => { let _ = app_for_thread.emit("token", format!("\n[ERR] forward(step@{seqlen_offset}): {e}")); break; }
      };

      let mut last = match pick_last_step(&logits) {
        Ok(t) => t, Err(e) => { let _ = app_for_thread.emit("token", format!("\n[ERR] index(step): {e}")); break; }
      };
      last = match last.to_dtype(DType::F32) {
        Ok(t) => t, Err(e) => { let _ = app_for_thread.emit("token", format!("\n[ERR] cast(step): {e}")); break; }
      };
      if let Ok(t) = last.clamp(-50f32, 50f32) { last = t; }

      next_id = match sampler.sample(&last) {
        Ok(id) => id as u32,
        Err(_) => match last.argmax(0)
          .and_then(|t| t.to_scalar::<u32>().or_else(|_| t.to_scalar::<i64>().map(|v| v as u32))) {
          Ok(v) => v, Err(_) => 0
        }
      };

      // stop on eos / </think>
      let vocab = tok.get_vocab(true);
      let eos_id = vocab.get("</s>").copied()
        .or_else(|| vocab.get("<|endoftext|>").copied());
      let end_think = vocab.get("</think>").copied();
      if eos_id == Some(next_id) || end_think == Some(next_id) { break; }

      // stream token
      let piece = tok.decode(&[next_id], false).unwrap_or_default();
      if piece.is_empty() {
        let _ = app_for_thread.emit("token", format!("[id:{next_id}]"));
      } else {
        let _ = app_for_thread.emit("token", piece);
      }
      n_tokens += 1;

      ids_u32.push(next_id);
      ids_i64.push(next_id as i64);
      seqlen_offset += 1;
    }

    // timing
    let dt = t0.elapsed().as_secs_f32();
    let _ = app_for_thread.emit(
      "token",
      format!("\n✓ done  ({} tok, {:.2}s, {:.2} tok/s)", n_tokens, dt, n_tokens as f32 / dt.max(1e-6))
    );
  });

  Ok(())
}

// ---------- boot ----------

fn main() {
  tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![echo, init_model, generate])
    .run(tauri::generate_context!())
    .expect("error while running tauri app");
}