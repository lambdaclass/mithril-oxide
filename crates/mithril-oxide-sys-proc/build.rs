use std::env::var;

fn main() {
    println!("cargo:rustc-env=PROFILE={}", var("PROFILE").unwrap());
}
