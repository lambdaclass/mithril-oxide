fn main() {
    println!(
        "cargo:rustc-link-search={}/lib",
        env!("MLIR_SYS_160_PREFIX")
    );
    println!("cargo:rustc-link-lib=LLVM");
    println!("cargo:rustc-link-lib=MLIR");
}
