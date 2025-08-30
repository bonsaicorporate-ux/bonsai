#![allow(clippy::needless_return)]

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use tauri::{AppHandle, Manager};
use tauri::Emitter;
const HOST: &str = "127.0.0.1";
const PORT: u16 = 8080; // llama-server port
// const MODEL_FILE: &str = "model-Q4_K_M.gguf"; // under resources/deepseek-model/
const MODEL_FILE: &str = "Phi-4-mini-reasoning-Q4_K_M.gguf";
const DEBUG_LLAMA_LOGS: bool = false; // flip to true if you want to see server logs
// -------- System primer you can edit any time --------
const USE_PRIMER: bool = true;
const SYSTEM_PRIMER: &str = "\
You are Phi, an expert math problem solver. Reason internally and do not reveal \
your scratch work. Provide only the final answer, ideally as \\boxed{...}. \
If the user explicitly asks for steps, give a brief, clean outline only.";
// --------------------------- App state ---------------------------

#[derive(Clone, Default)]
struct AppState {
  server_url: Arc<std::sync::Mutex<String>>,
}

#[tauri::command]
fn echo(text: String) -> String {
  text
}

#[tauri::command]
fn init_model(app: tauri::AppHandle) -> Result<String, String> {
    let url = format!("http://{}:{}", HOST, PORT);

    // ✅ put the lock in its own scope so the guard drops before we do other work
    {
      let state = app.state::<AppState>();
      if let Ok(mut s) = state.server_url.lock() {
        *s = url.clone();
      }
    }
  // Ensure server is running (start if needed)
  if let Err(e) = start_llama_if_needed(&app) {
    let _ = app.emit("token", format!("[ERR] start llama: {e}"));
    return Err(e);
  }

  // Wait until it’s actually accepting connections
  if !wait_for_server_ready(30) {
    return Err(format!("llama-server at {} not healthy in time", url));
  }

  Ok("ready".into())
}

// --------------------- llama-server bootstrap --------------------

fn resources_dir(app: &AppHandle) -> Result<PathBuf, String> {
  app
    .path()
    .resolve("resources", tauri::path::BaseDirectory::Resource)
    .map_err(|e| format!("resolve resources/: {e}"))
}

fn server_bin(app: &AppHandle) -> Result<PathBuf, String> {
  Ok(resources_dir(app)?.join("llama-bin/llama-server"))
}

fn model_path(app: &AppHandle) -> Result<PathBuf, String> {
  Ok(resources_dir(app)?.join(format!("deepseek-model/{MODEL_FILE}")))
}

fn server_url() -> String {
  format!("http://{HOST}:{PORT}")
}

fn server_healthy() -> bool {
    // simple TCP connect check instead of HTTP GET to /health
    use std::net::TcpStream;
    let addr = format!("{}:{}", HOST, PORT);
    TcpStream::connect(addr).is_ok()
  }

fn wait_for_server_ready(timeout_secs: u64) -> bool {
  let start = std::time::Instant::now();
  while start.elapsed().as_secs() < timeout_secs {
    if server_healthy() {
      return true;
    }
    std::thread::sleep(std::time::Duration::from_millis(200));
  }
  false
}

fn ensure_executable(p: &Path) -> Result<(), String> {
  #[cfg(target_os = "macos")]
  {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(p)
      .map_err(|e| format!("metadata {}: {e}", p.display()))?
      .permissions();
    // rwx for user
    perms.set_mode(perms.mode() | 0o700);
    std::fs::set_permissions(p, perms)
      .map_err(|e| format!("chmod {}: {e}", p.display()))?;
    // best effort: strip quarantine if present (ignore errors)
    let _ = Command::new("/usr/bin/xattr")
      .args(["-dr", "com.apple.quarantine", p.to_string_lossy().as_ref()])
      .output();
  }
  Ok(())
}

fn start_llama_if_needed(app: &AppHandle) -> Result<(), String> {
  if server_healthy() {
    let _ = app.emit("token", "[llama] already running");
    return Ok(());
  }

  let bin = server_bin(app)?;
  let model = model_path(app)?;

  if !bin.exists() {
    return Err(format!("llama-server missing at {}", bin.display()));
  }
  if !model.exists() {
    return Err(format!("model missing at {}", model.display()));
  }

  ensure_executable(&bin)?;

  // spawn the server
  let mut child = Command::new(&bin)
    .args([
      "-m", &model.to_string_lossy(),
      "--host", HOST,
      "--port", &PORT.to_string(),
      "-ngl", "999",
      "-t", "8",
    ])
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .map_err(|e| format!("spawn llama-server: {e}"))?;

  // Stream STDOUT/STDERR to UI
  if let Some(out) = child.stdout.take() {
    let app_clone = app.clone();
    std::thread::spawn(move || {
      use std::io::{BufRead, BufReader};
      let reader = BufReader::new(out);
      for line in reader.lines().flatten() {
        if DEBUG_LLAMA_LOGS {
            let _ = app_clone.emit("log", format!("[llama] {line}"));
          }
      }
    });
  }
  if let Some(err) = child.stderr.take() {
    let app_clone = app.clone();
    std::thread::spawn(move || {
      use std::io::{BufRead, BufReader};
      let reader = BufReader::new(err);
      for line in reader.lines().flatten() {
        if DEBUG_LLAMA_LOGS {
            let _ = app_clone.emit("log", format!("[llama!] {line}"));
          }
      }
    });
  }

  // detach child (we don’t wait here)
  std::mem::forget(child);

  // wait for readiness
  if wait_for_server_ready(20) {
    if DEBUG_LLAMA_LOGS {
        let _ = app.emit("log", "[llama] ready on :8080");
      }
    Ok(())
  } else {
    Err("llama-server did not become healthy in time".into())
  }
}

// --------------------------- Completion API ---------------------------

#[derive(Debug, Serialize)]
struct CompletionRequest<'a> {
  prompt: &'a str,
  #[serde(skip_serializing_if = "Option::is_none")]
  stop: Option<Vec<&'a str>>,
  #[serde(skip_serializing_if = "Option::is_none")]
  temperature: Option<f32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  top_p: Option<f32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  n_predict: Option<i32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  cache_prompt: Option<bool>,
  stream: bool,
}

#[derive(Debug, Deserialize)]
struct CompletionChunk {
  #[serde(default)]
  content: String,
  #[serde(default)]
  stop: bool,
}

#[tauri::command]
fn generate(app: tauri::AppHandle, prompt: String) -> Result<(), String> {
    let user = prompt.trim();
    let final_prompt = format!(
        "<|system|>{sys}<|end|><|user|>{u}<|end|><|assistant|>",
        sys = SYSTEM_PRIMER,
        u = user
    );

  let app_for_thread = app.clone();
  tauri::async_runtime::spawn(async move {
    use futures_util::StreamExt;
    use tauri::Emitter;

    // ensure URL in state
    let state: tauri::State<'_, AppState> = app_for_thread.state();
    let server_url = {
      match state.server_url.lock() {
        Ok(u) => u.clone(),
        Err(_) => {
          let _ = app_for_thread.emit("token", "[ERR] server_url lock");
          return;
        }
      }
    };

    let url = format!("{}/completion", server_url);
    let _ = app_for_thread.emit("token", "▶︎ start ");

    let req_body = CompletionRequest {
        prompt: &final_prompt,
        stop: Some(vec!["<|end|>"]),   // Phi chat stop
        temperature: Some(0.7),        // math-friendly
        top_p: Some(0.95),
        n_predict: Some(2048),         // give it room; raise if you need longer
        cache_prompt: Some(true),
        stream: true,
      };

    let client = match reqwest::Client::builder().build() {
      Ok(c) => c,
      Err(e) => {
        let _ = app_for_thread.emit("token", format!("[ERR] build client: {e}"));
        return;
      }
    };

    let resp = match client.post(&url).json(&req_body).send().await {
        Ok(r) => r,
        Err(e) => {
          use std::error::Error as _;
          let mut msg = format!("[ERR] POST {url}: {e}");
          let mut src = e.source();
          while let Some(s) = src {
            msg.push_str(&format!(" :: {}", s));
            src = s.source();
          }
          let _ = app_for_thread.emit("token", msg);
          return;
        }
      };

    if !resp.status().is_success() {
      let code = resp.status();
      let text = resp.text().await.unwrap_or_default();
      let _ = app_for_thread.emit("token", format!("[ERR] HTTP {code} {text}"));
      return;
    }


    // --- hide <think> until </think> ---
    let mut hide_think = true;
    let mut stash = String::new();

    // stream SSE-like lines
    let mut stream = resp.bytes_stream();
    let mut buf = Vec::<u8>::new();

    while let Some(chunk) = stream.next().await {
      match chunk {
        Ok(bytes) => {
          buf.extend_from_slice(&bytes);
          while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line = buf.drain(..=pos).collect::<Vec<_>>();
            if line.starts_with(b"data: ") {
              let payload = &line[6..line.len().saturating_sub(1)];
              if payload == b"[DONE]" {
                let _ = app_for_thread.emit("token", "\n✓ done");
                return;
              }
              if let Ok(text) = std::str::from_utf8(payload) {
                match serde_json::from_str::<CompletionChunk>(text) {
                  Ok(ch) => {
                    let piece = ch.content;
                    if !piece.trim().is_empty() {
                        if hide_think {
                          stash.push_str(&piece);
                          if let Some(idx) = stash.find("</think>") {
                            // everything after </think> is revealed
                            let after = stash[idx + "</think>".len()..].to_string();
                            if !after.trim().is_empty() {
                              let _ = app_for_thread.emit("token", after);
                            }
                            hide_think = false;
                            stash.clear();
                          }
                        } else {
                          let _ = app_for_thread.emit("token", piece);
                        }
                      }
                    if ch.stop {
                      let _ = app_for_thread.emit("token", "\n✓ done");
                      return;
                    }
                  }
                  Err(e) => {
                    let _ = app_for_thread.emit(
                      "token",
                      format!("\n[WARN] bad chunk: {e} :: {text}"),
                    );
                  }
                }
              }
            }
          }
        }
        Err(e) => {
          let _ = app_for_thread.emit("token", format!("\n[ERR] stream: {e}"));
          break;
        }
      }
    }

    let _ = app_for_thread.emit("token", "\n✓ done");
  });

  Ok(())
}

// ------------------------------ Boot ------------------------------

fn main() {
  tauri::Builder::default()
    .manage(AppState::default())
    .setup(|app| {
      // OWN the handle so it’s 'static inside the thread
      let app_handle = app.handle().clone();
      std::thread::spawn(move || {
        if let Err(e) = start_llama_if_needed(&app_handle) {
          let _ = app_handle.emit("token", format!("[ERR] start llama: {e}"));
        }
      });
      Ok(())
    })
    .invoke_handler(tauri::generate_handler![echo, init_model, generate])
    .run(tauri::generate_context!())
    .expect("error while running tauri app");
}