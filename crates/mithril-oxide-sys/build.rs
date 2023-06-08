use std::env::var;

fn main() {
    println!(
        "cargo:rustc-link-search={}/lib",
        env!("MLIR_SYS_160_PREFIX")
    );
    println!("cargo:rustc-link-search={}", var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-lib=LLVM");
    println!("cargo:rustc-link-lib=MLIR");
}
