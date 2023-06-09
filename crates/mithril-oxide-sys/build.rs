use std::{
    env::var,
    io::Read,
    path::Path,
    process::{Command, Stdio},
};

fn main() {
    let mlir_env = env!("MLIR_SYS_160_PREFIX");
    println!("cargo:rustc-link-search={}/lib", mlir_env);
    println!("cargo:rustc-link-search={}", var("OUT_DIR").unwrap());

    // Forward a path to the auxiliary library, and link to it globally.
    println!(
        "cargo:rustc-env=AUXLIB_PATH={}/libauxlib.a",
        var("OUT_DIR").unwrap()
    );
    println!("cargo:rustc-link-search={}", var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-lib=auxlib");
    println!("cargo:rustc-link-lib=stdc++");

    // linking to llvm here works if llvm-config --shared outputs shared, not static, because in static mlir likely includes llvm too.
    let mut process = Command::new(Path::new(mlir_env).join("bin/llvm-config"))
        .arg("--shared-mode")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    process.wait().unwrap();

    let mut output = String::new();
    process.stdout.unwrap().read_to_string(&mut output).unwrap();
    let output = output.trim();

    match output {
        "static" => {}
        "shared" => println!("cargo:rustc-link-lib=LLVM"),
        _ => panic!("unknown shared mode: {}", output),
    }
    println!("cargo:rustc-link-lib=MLIR");
}
