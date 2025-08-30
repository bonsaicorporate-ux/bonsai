fn main() {
    // make sure changes to tauri.conf.json re-run the build script
    println!("cargo:rerun-if-env-changed=TAURI_CONFIG");

    // run tauri’s codegen
    tauri_build::build();

    // NEW: satisfy tauri-build’s expectation *and* Cargo’s format
    println!("cargo:dev=true");
}