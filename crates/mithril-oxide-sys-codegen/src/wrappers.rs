use std::{
    env::{current_dir, var},
    ffi::OsStr,
    io::{BufRead, Cursor},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// Find the path to `llvm-config`.
fn find_llvm_config() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(var("MLIR_SYS_160_PREFIX")?).join("bin/llvm-config"))
}

/// Find the path to `llvm-ar`.
fn find_ar() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(var("MLIR_SYS_160_PREFIX")?).join("bin/llvm-ar"))
}

/// Find the path to `clang++`.
fn find_clang() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(var("MLIR_SYS_160_PREFIX")?).join("bin/clang++"))
}

/// Find the standard clang include paths, as well as those for LLVM and MLIR.
pub fn extract_clang_include_paths(path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let process = Command::new(find_clang()?)
        .arg("-c")
        .arg("-v")
        .arg(path)
        .args(["-o", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;

    let output = process.wait_with_output()?;
    let mut stderr = Cursor::new(output.stderr);

    let mut buffer = String::new();
    while stderr.read_line(&mut buffer)? != 0 && buffer != "#include \"...\" search starts here:\n"
    {
        buffer.clear();
    }

    let mut include_paths = Vec::new();
    while stderr.read_line(&mut buffer)? != 0 && buffer != "End of search list.\n" {
        if buffer.starts_with("#include") && buffer.ends_with("search starts here:\n") {
            buffer.clear();
            continue;
        }

        let include_path = buffer.trim();
        include_paths.push(include_path.to_string());

        buffer.clear();
    }
    buffer.clear();

    // Append paths from `llvm-config`.
    let process = Command::new(find_llvm_config()?)
        .arg("--includedir")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    let output = process.wait_with_output()?;
    let mut stdout = Cursor::new(output.stdout);
    while stdout.read_line(&mut buffer)? != 0 {
        let include_path = buffer.trim();
        include_paths.push(include_path.to_string());
        buffer.clear();
    }

    // Append current directory (should be the workspace root).
    include_paths.push(current_dir().unwrap().to_string_lossy().into_owned());

    Ok(include_paths)
}

/// Builds the auxiliary library into an archive file (a static library).
pub fn build_auxiliary_library(
    target_path: &Path,
    source_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(source_path.extension().and_then(OsStr::to_str), Some("cpp"));
    assert_eq!(target_path.extension().and_then(OsStr::to_str), Some("a"));

    // Debugging instructions:
    //   - Uncomment the line after this comment to display the generated C++.
    //   - Replace `Stdio::null()` by `Stdio::inherit()` in the command below.
    //   - Remember to leave as it is now just in case (the process's i/o is already used for
    //     transferring data and clang may break stuff if not filtered out).
    //
    // eprintln!("{}", std::fs::read_to_string(source_path).unwrap());

    let mut process = Command::new(find_clang()?)
        .arg("-c")
        .arg("-std=c++17")
        .arg(source_path.to_string_lossy().as_ref())
        .args(
            &extract_clang_include_paths(source_path)?
                .into_iter()
                .map(|x| format!("-I{x}"))
                .collect::<Vec<_>>(),
        )
        .arg(&format!("-o{}", source_path.with_extension("o").display()))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    assert!(process.wait()?.success());

    let mut process = Command::new(find_ar()?)
        .arg("crs")
        .arg(target_path.to_string_lossy().as_ref())
        .arg(source_path.with_extension("o").to_string_lossy().as_ref())
        .spawn()?;
    assert!(process.wait()?.success());

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs;

    #[test]
    fn test_find_clang() {
        assert!(find_clang().unwrap().exists());
    }

    #[test]
    fn test_extract_clang_include_paths() {
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_cpp = temp_dir.path().join("source.cpp");
        fs::write(&temp_cpp, b"").unwrap();

        let include_paths = extract_clang_include_paths(&temp_cpp).unwrap();
        assert!(!include_paths.is_empty());
    }
}
