#![allow(clippy::needless_return)]

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tauri::Manager;

const SERVER_URL: &str = "http://127.0.0.1:8080"; // llama.cpp server

#[derive(Clone, Default)]
struct AppState {
  server_url: Arc<Mutex<String>>,
}

#[tauri::command]
fn echo(text: String) -> String {
  text
}

#[tauri::command]
fn init_model(app: tauri::AppHandle) -> Result<String, String> {
  // You can make this configurable if you want.
  let state: tauri::State<'_, AppState> = app.state();
  if let Ok(mut url) = state.server_url.lock() {
    *url = SERVER_URL.to_string();
  }
  Ok("ready".into())
}

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
  // llama.cpp /completion stream chunks typically include these:
  #[serde(default)]
  content: String,
  #[serde(default)]
  stop: bool,
  // sometimes include timings / tokens, we ignore safely if absent
}

#[tauri::command]
fn generate(app: tauri::AppHandle, prompt: String) -> Result<(), String> {
  // The DeepSeek-R1 readme recommends starting output with <think>\n (no system prompt).
  let user_prompt = prompt.trim().to_string();
  let final_prompt = format!("{user}\n<think>\n", user = user_prompt);

  // background task: async streaming via reqwest
  let app_for_thread = app.clone();
  tauri::async_runtime::spawn(async move {
    use futures_util::StreamExt;
    use tauri::Emitter;

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

    // we’ll stop if model emits </think> (recommended by R1 notes)
    let req_body = CompletionRequest {
      prompt: &final_prompt,
      stop: Some(vec!["</think>"]),
      temperature: Some(0.6),  // per R1 rec: 0.5–0.7
      top_p: Some(0.95),
      n_predict: Some(256),     // adjust as you like
      cache_prompt: Some(true),
      stream: true,
    };

    let client = reqwest::Client::new();
    let resp = match client.post(&url).json(&req_body).send().await {
      Ok(r) => r,
      Err(e) => {
        let _ = app_for_thread.emit("token", format!("[ERR] POST {url}: {e}"));
        return;
      }
    };

    if !resp.status().is_success() {
      let code = resp.status();
      let text = resp.text().await.unwrap_or_default();
      let _ = app_for_thread.emit("token", format!("[ERR] HTTP {code}: {text}"));
      return;
    }

    // The stream is SSE-like: lines beginning with "data: {...}"
    let mut stream = resp.bytes_stream();

    // simple line buffer across chunks
    let mut buf = Vec::<u8>::new();

    while let Some(chunk) = stream.next().await {
      match chunk {
        Ok(bytes) => {
          buf.extend_from_slice(&bytes);

          // split by '\n'
          while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line = buf.drain(..=pos).collect::<Vec<_>>();
            if line.starts_with(b"data: ") {
              let payload = &line[6..line.len().saturating_sub(1)]; // strip "data: " and trailing '\n'
              if payload == b"[DONE]" {
                // finished
                let _ = app_for_thread.emit("token", "\n✓ done");
                return;
              }
              // try to parse JSON chunk
              if let Ok(text) = std::str::from_utf8(payload) {
                match serde_json::from_str::<CompletionChunk>(text) {
                  Ok(ch) => {
                    if !ch.content.is_empty() {
                      let _ = app_for_thread.emit("token", ch.content);
                    }
                    if ch.stop {
                      let _ = app_for_thread.emit("token", "\n✓ done");
                      return;
                    }
                  }
                  Err(e) => {
                    // non-fatal: print raw for debugging
                    let _ = app_for_thread.emit("token", format!("\n[WARN] bad chunk: {e} :: {text}"));
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

fn main() {
  tauri::Builder::default()
    .manage(AppState::default())
    .invoke_handler(tauri::generate_handler![echo, init_model, generate])
    .run(tauri::generate_context!())
    .expect("error while running tauri app");
}