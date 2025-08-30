#![allow(clippy::needless_return)]

use serde::{Deserialize, Serialize};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::io::{BufRead, BufReader};
use tauri::{AppHandle, Manager};
use tauri::Emitter;
use tokio::sync::Mutex;

const HOST: &str = "127.0.0.1";
const MODEL_FILE: &str = "Phi-4-mini-reasoning-Q4_K_M.gguf";
const DEBUG_LLAMA_LOGS: bool = false;

// System primer
const USE_PRIMER: bool = true;
const SYSTEM_PRIMER: &str = "\
You are Phi, an expert math problem solver. Reason internally and do not reveal \
your scratch work. Provide only the final answer, ideally as \\boxed{...}. \
If the user explicitly asks for steps, give a brief, clean outline only.";

// --------------------------- App state ---------------------------

#[derive(Clone)]
struct LlamaServerHandle {
    port: u16,
    process_id: u32,
    base_url: String,
}

#[derive(Default)]
struct AppState {
    server_handle: Arc<Mutex<Option<LlamaServerHandle>>>,
    server_process: Arc<Mutex<Option<Child>>>,
}

#[tauri::command]
fn echo(text: String) -> String {
    text
}

#[tauri::command]
async fn init_model(app: tauri::AppHandle) -> Result<String, String> {
    let state: tauri::State<'_, AppState> = app.state();
    let handle = state.server_handle.lock().await;
    
    if let Some(h) = handle.as_ref() {
        Ok(format!("ready on port {}", h.port))
    } else {
        Err("Server not initialized".into())
    }
}

#[tauri::command]
async fn get_server_status(app: tauri::AppHandle) -> Result<String, String> {
    let state: tauri::State<'_, AppState> = app.state();
    let mut process_guard = state.server_process.lock().await;
    
    if let Some(ref mut child) = *process_guard {
        match child.try_wait() {
            Ok(None) => Ok("running".into()),
            Ok(Some(status)) => Ok(format!("exited: {}", status)),
            Err(e) => Err(format!("error checking status: {}", e))
        }
    } else {
        Ok("not started".into())
    }
}

// --------------------- Port management --------------------

fn find_available_port() -> Result<u16, String> {
    // Try a range of preferred ports first
    let preferred_ports = vec![8080, 8081, 8082, 11434, 3000, 3001, 5000, 5001];
    
    for port in preferred_ports {
        if is_port_available(port) {
            return Ok(port);
        }
    }
    
    // Fall back to letting OS assign a port
    let listener = TcpListener::bind(format!("{}:0", HOST))
        .map_err(|e| format!("Failed to bind to any port: {}", e))?;
    
    let port = listener.local_addr()
        .map_err(|e| format!("Failed to get local addr: {}", e))?
        .port();
    
    // Explicitly drop the listener to free the port
    drop(listener);
    
    // Small delay to ensure OS has released the port
    std::thread::sleep(std::time::Duration::from_millis(100));
    
    Ok(port)
}

fn is_port_available(port: u16) -> bool {
    TcpListener::bind(format!("{}:{}", HOST, port)).is_ok()
}

// --------------------- llama-server bootstrap --------------------

fn resources_dir(app: &AppHandle) -> Result<PathBuf, String> {
    app
        .path()
        .resolve("resources", tauri::path::BaseDirectory::Resource)
        .map_err(|e| format!("resolve resources/: {e}"))
}

fn server_bin(app: &AppHandle) -> Result<PathBuf, String> {
    let bin_name = if cfg!(target_os = "windows") {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    Ok(resources_dir(app)?.join(format!("llama-bin/{}", bin_name)))
}

fn model_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(resources_dir(app)?.join(format!("deepseek-model/{}", MODEL_FILE)))
}

fn server_healthy_on_port(port: u16) -> bool {
    use std::net::TcpStream;
    use std::time::Duration;
    
    let addr = format!("{}:{}", HOST, port);
    match std::net::TcpStream::connect_timeout(
        &addr.parse().unwrap_or_else(|_| format!("{}:{}", HOST, port).parse().unwrap()),
        Duration::from_secs(1)
    ) {
        Ok(_) => true,
        Err(_) => false
    }
}

async fn wait_for_server_ready(port: u16, timeout_secs: u64) -> bool {
    let start = std::time::Instant::now();
    
    while start.elapsed().as_secs() < timeout_secs {
        if server_healthy_on_port(port) {
            // Try a health check via HTTP as well
            if let Ok(true) = check_health_endpoint(port).await {
                return true;
            }
            // Fall back to just TCP connection being available
            return true;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }
    false
}

async fn check_health_endpoint(port: u16) -> Result<bool, String> {
    let health_url = format!("http://{}:{}/health", HOST, port);
    
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .map_err(|e| format!("Failed to build client: {}", e))?;
    
    match client.get(&health_url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                let body = response.text().await.unwrap_or_default();
                Ok(body.contains("ok") || body.contains("healthy"))
            } else {
                Ok(false)
            }
        }
        Err(_) => Ok(false)
    }
}

fn ensure_executable(p: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        
        let mut perms = std::fs::metadata(p)
            .map_err(|e| format!("metadata {}: {e}", p.display()))?
            .permissions();
        
        // rwx for user
        perms.set_mode(perms.mode() | 0o700);
        std::fs::set_permissions(p, perms)
            .map_err(|e| format!("chmod {}: {e}", p.display()))?;
        
        // Strip quarantine on macOS
        #[cfg(target_os = "macos")]
        {
            let _ = Command::new("/usr/bin/xattr")
                .args(["-dr", "com.apple.quarantine", p.to_string_lossy().as_ref()])
                .output();
        }
    }
    
    Ok(())
}

async fn kill_orphaned_servers() {
    #[cfg(unix)]
    {
        // Try to kill any existing llama-server processes
        let _ = Command::new("pkill")
            .args(["-f", "llama-server"])
            .output();
    }
    
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("taskkill")
            .args(["/F", "/IM", "llama-server.exe"])
            .output();
    }
    
    // Give OS time to clean up
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
}

async fn start_llama_server(app: &AppHandle) -> Result<(Child, u16, String), String> {
    let bin = server_bin(app)?;
    let model = model_path(app)?;
    
    if !bin.exists() {
        return Err(format!("llama-server missing at {}", bin.display()));
    }
    if !model.exists() {
        return Err(format!("model missing at {}", model.display()));
    }
    
    ensure_executable(&bin)?;
    
    // Find available port
    let port = find_available_port()?;
    let server_url = format!("http://{}:{}", HOST, port);
    
    let _ = app.emit("log", format!("[llama] Starting on port {}", port));
    
    // Prepare the command
    let mut cmd = Command::new(&bin);
    
    cmd.args([
        "-m", &model.to_string_lossy(),
        "--host", HOST,
        "--port", &port.to_string(),
        "-ngl", "999",
        "-t", "8",
        "--ctx-size", "4096",
        "--no-mmap",  // Can help with stability
    ])
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .stdin(Stdio::null());
    
    // Set working directory to resources
    if let Ok(res_dir) = resources_dir(app) {
        cmd.current_dir(res_dir);
    }
    
    // Clean environment on macOS
    #[cfg(target_os = "macos")]
    {
        cmd.env_remove("DYLD_LIBRARY_PATH");
        cmd.env_remove("DYLD_FALLBACK_LIBRARY_PATH");
    }
    
    // Spawn the process
    let mut child = cmd.spawn()
        .map_err(|e| format!("spawn llama-server: {e}"))?;
    
    let process_id = child.id();
    
    // Stream STDOUT
    if let Some(out) = child.stdout.take() {
        let app_clone = app.clone();
        std::thread::spawn(move || {
            let reader = BufReader::new(out);
            for line in reader.lines().flatten() {
                if DEBUG_LLAMA_LOGS {
                    let _ = app_clone.emit("log", format!("[llama] {}", line));
                }
            }
        });
    }
    
    // Stream STDERR
    if let Some(err) = child.stderr.take() {
        let app_clone = app.clone();
        std::thread::spawn(move || {
            let reader = BufReader::new(err);
            for line in reader.lines().flatten() {
                if DEBUG_LLAMA_LOGS || line.contains("error") || line.contains("ERROR") {
                    let _ = app_clone.emit("log", format!("[llama!] {}", line));
                }
            }
        });
    }
    
    let _ = app.emit("log", format!("[llama] Process started with PID {}", process_id));
    
    Ok((child, port, server_url))
}

async fn start_llama_if_needed(app: &AppHandle) -> Result<(), String> {
    let state: tauri::State<'_, AppState> = app.state();
    
    // Check if already running
    {
        let handle = state.server_handle.lock().await;
        if let Some(h) = handle.as_ref() {
            if server_healthy_on_port(h.port) {
                let _ = app.emit("log", format!("[llama] Already running on port {}", h.port));
                return Ok(());
            }
        }
    }
    
    // Kill any orphaned processes
    kill_orphaned_servers().await;
    
    // Try to start the server with retries
    let mut attempts = 0;
    const MAX_ATTEMPTS: u32 = 3;
    
    while attempts < MAX_ATTEMPTS {
        attempts += 1;
        
        let _ = app.emit("log", format!("[llama] Start attempt {}/{}", attempts, MAX_ATTEMPTS));
        
        match start_llama_server(app).await {
            Ok((child, port, url)) => {
                // Wait for server to be ready
                if wait_for_server_ready(port, 30).await {
                    // Store the handle
                    let handle = LlamaServerHandle {
                        port,
                        process_id: child.id(),
                        base_url: url.clone(),
                    };
                    
                    *state.server_handle.lock().await = Some(handle);
                    *state.server_process.lock().await = Some(child);
                    
                    let _ = app.emit("log", format!("[llama] Ready on port {}", port));
                    return Ok(());
                } else {
                    // Server didn't become ready, kill it
                    if let Some(mut child) = state.server_process.lock().await.take() {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                    
                    if attempts < MAX_ATTEMPTS {
                        let _ = app.emit("log", "[llama] Server didn't become ready, retrying...");
                        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    }
                }
            }
            Err(e) => {
                let _ = app.emit("log", format!("[llama] Failed to start: {}", e));
                if attempts < MAX_ATTEMPTS {
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }
            }
        }
    }
    
    Err(format!("Failed to start llama-server after {} attempts", MAX_ATTEMPTS))
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
async fn generate(app: tauri::AppHandle, prompt: String) -> Result<(), String> {
    let user = prompt.trim();
    let final_prompt = if USE_PRIMER {
        format!(
            "<|system|>{}<|end|><|user|>{}<|end|><|assistant|>",
            SYSTEM_PRIMER, user
        )
    } else {
        format!("<|user|>{}<|end|><|assistant|>", user)
    };
    
    // Get server URL
    let state: tauri::State<'_, AppState> = app.state();
    let server_url = {
        let handle = state.server_handle.lock().await;
        match handle.as_ref() {
            Some(h) => h.base_url.clone(),
            None => {
                let _ = app.emit("token", "[ERR] Server not initialized");
                return Err("Server not initialized".into());
            }
        }
    };
    
    // Check if server is still running
    {
        let mut process_guard = state.server_process.lock().await;
        if let Some(ref mut child) = *process_guard {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let _ = app.emit("token", format!("[ERR] Server died: {}", status));
                    // Try to restart
                    drop(process_guard);
                    start_llama_if_needed(&app).await?;
                }
                Ok(None) => {}, // Still running
                Err(e) => {
                    let _ = app.emit("log", format!("[WARN] Can't check server status: {}", e));
                }
            }
        }
    }
    
    let app_for_thread = app.clone();
    
    tauri::async_runtime::spawn(async move {
        use futures_util::StreamExt;
        
        let url = format!("{}/completion", server_url);
        let _ = app_for_thread.emit("token", "▶︎ start ");
        
        let req_body = CompletionRequest {
            prompt: &final_prompt,
            stop: Some(vec!["<|end|>"]),
            temperature: Some(0.7),
            top_p: Some(0.95),
            n_predict: Some(2048),
            cache_prompt: Some(true),
            stream: true,
        };
        
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build() {
            Ok(c) => c,
            Err(e) => {
                let _ = app_for_thread.emit("token", format!("[ERR] build client: {e}"));
                return;
            }
        };
        
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
            let _ = app_for_thread.emit("token", format!("[ERR] HTTP {code} {text}"));
            return;
        }
        
        // Hide <think> until </think>
        let mut hide_think = true;
        let mut stash = String::new();
        
        // Stream SSE-like lines
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

// ------------------------------ Cleanup ------------------------------

#[tauri::command]
async fn shutdown_server(app: tauri::AppHandle) -> Result<String, String> {
    let state: tauri::State<'_, AppState> = app.state();
    
    let mut process_guard = state.server_process.lock().await;
    if let Some(mut child) = process_guard.take() {
        // Try graceful shutdown first
        #[cfg(unix)]
        {
            use nix::sys::signal::{self, Signal};
            use nix::unistd::Pid;
            
            if let Ok(pid) = i32::try_from(child.id()) {
                let _ = signal::kill(Pid::from_raw(pid), Signal::SIGTERM);
            }
        }
        
        // Give it time to shut down gracefully
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        
        // Force kill if still running
        match child.try_wait() {
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
            }
            _ => {}
        }
        
        *state.server_handle.lock().await = None;
        
        Ok("Server shut down".into())
    } else {
        Ok("Server was not running".into())
    }
}

// ------------------------------ Boot ------------------------------

fn main() {
    tauri::Builder::default()
        .manage(AppState::default())
        .setup(|app| {
            let app_handle = app.handle().clone();
            
            // Start server in async context
            tauri::async_runtime::spawn(async move {
                if let Err(e) = start_llama_if_needed(&app_handle).await {
                    let _ = app_handle.emit("token", format!("[ERR] start llama: {e}"));
                }
            });
            
            Ok(())
        })
        .on_window_event(|event| {
            // Clean shutdown on window close
            if let tauri::WindowEvent::CloseRequested { .. } = event.event() {
                let app = event.window().app_handle().clone();
                tauri::async_runtime::block_on(async move {
                    let _ = shutdown_server(app).await;
                });
            }
        })
        .invoke_handler(tauri::generate_handler![
            echo,
            init_model,
            generate,
            get_server_status,
            shutdown_server
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri app");
}