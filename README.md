# Mithril Oxide

Rust bindings to MLIR via the C++ API.


## Project structure

  - `mithril-oxide`: Our Rusty bindings.
  - `mithril-oxide-sys`: Raw unsafe bindings to the C++ API.
  - `mithril-oxide-sys-codegen`: Automated binding generation (actual type checking and code
    generation).
  - `mithril-oxide-sys-proc`: Automated binding generation (via proc macros).


## Compilation instructions

There'll have to be some environment variables set; in particular:

```bash
# Path to `llvm-config`. Can be ignored if it's available on `PATH`.
export LLVM_CONFIG_PATH=/usr/lib/llvm-16/bin/llvm-config

# Path to where `libclang.so` resides. Depending on the installation method, you may need to make a
# symbolic link for it to be detectable.
export LIBCLANG_PATH=/usr/lib/llvm-16/lib

# To execute, the library loader will need to find `libLLVM.so` and `libMLIR.so`.
export LD_LIBRARY_PATH=/usr/lib/llvm-16/lib
```

The commands to build the project are as follow:

```bash
# The code generation binary must be available to build the project.
cargo build -p mithril-oxide-sys-codegen

# Build the project.
cargo build

# Run an example (or not).
cargo run --example hello
```
