use std::env::var;

fn main() {
    println!(
        "cargo:rustc-link-search={}/lib",
        env!("MLIR_SYS_160_PREFIX")
    );
    println!("cargo:rustc-link-search={}", var("OUT_DIR").unwrap());
    // linking to llvm here works if llvm-config --shared outputs shared, not static, because in static mlir likely includes llvm too.
    // println!("cargo:rustc-link-lib=LLVM");
    println!("cargo:rustc-link-lib=MLIR");
}
