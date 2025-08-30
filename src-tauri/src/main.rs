use tauri::Manager;

#[tauri::command]
fn echo(text: String) -> String {
    text
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![echo])
        .run(tauri::generate_context!())
        .expect("error while running app");
}