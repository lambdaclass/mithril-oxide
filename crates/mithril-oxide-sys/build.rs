fn main() {
    println!("cargo:rustc-link-search=/usr/lib/llvm-16/lib");
    println!("cargo:rustc-link-lib=LLVM");
    println!("cargo:rustc-link-lib=MLIR");
}
