#![deny(clippy::pedantic)]
#![deny(warnings)]

use std::{
    collections::BTreeSet,
    env::var,
    ffi::OsStr,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// C++ source code prefix.
const CPP_PREFIX: &str = "cpp";
/// Rust source code prefix.
const SRC_PREFIX: &str = "src";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (cpp_sources, src_sources) = {
        let all_sources = find_sources()?;

        (
            all_sources
                .iter()
                .map(|path| Path::new(CPP_PREFIX).join(path).with_extension("cpp"))
                .collect::<Vec<_>>(),
            all_sources
                .iter()
                .map(|path| Path::new(SRC_PREFIX).join(path).with_extension("rs"))
                .collect::<Vec<_>>(),
        )
    };

    cxx_build::bridges(src_sources)
        .files(&cpp_sources)
        .flag("-std=c++17")
        .flag(&format!("-I{}/include", var("MLIR_SYS_160_PREFIX")?))
        .flag("-Wno-comment")
        .flag("-Wno-unused-parameter")
        .compile("mlir-sys");

    // Build script re-run conditions.
    println!("cargo:rerun-if-changed={CPP_PREFIX}");
    //for src_path in &src_sources {
    //    println!("cargo:rerun-if-changed={}", src_path.to_str().unwrap());
    //}

    // Linker flags.
    println!(
        "cargo:rustc-link-search={}/lib",
        var("MLIR_SYS_160_PREFIX")?
    );
    if is_llvm_shared_mode()? {
        println!("cargo:rustc-link-lib=LLVM");
    }
    println!("cargo:rustc-link-lib=MLIR");

    Ok(())
}

/// Scan the C++ sources folder recursively for sources
fn find_sources() -> Result<BTreeSet<PathBuf>, Box<dyn std::error::Error>> {
    fn walk_dir(
        state: &mut BTreeSet<PathBuf>,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for dir_entry in path.as_ref().read_dir()? {
            let dir_entry = dir_entry?;

            let path = dir_entry.path();
            let kind = dir_entry.file_type()?;

            if kind.is_dir() {
                walk_dir(state, path)?;
            } else {
                // Skip non-C++ files.
                if path.extension().and_then(OsStr::to_str) != Some("cpp") {
                    continue;
                }

                // Insert the path without neither the C++ path prefix nor the C++ extension.
                state.insert(path.strip_prefix(CPP_PREFIX)?.with_extension(""));
            }
        }

        Ok(())
    }

    let mut state = BTreeSet::new();
    walk_dir(&mut state, CPP_PREFIX)?;

    Ok(state)
}

fn is_llvm_shared_mode() -> Result<bool, Box<dyn std::error::Error>> {
    let output =
        Command::new(Path::new(var("MLIR_SYS_160_PREFIX")?.as_str()).join("bin/llvm-config"))
            .arg("--shared-mode")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?
            .wait_with_output()?;

    Ok(match std::str::from_utf8(&output.stdout)?.trim() {
        "static" => false,
        "shared" => true,
        x => panic!("Invalid LLVM build mode: expected 'static' or 'shared' but instead got {x}"),
    })
}
