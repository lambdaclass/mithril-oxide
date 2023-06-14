# Mithril Oxide

_~Mithril~ MLIR is hard to rust but we did it anyway._

Rust bindings to MLIR via the C++ API.

## Project structure
  - `mithril-oxide`: Our Rusty bindings.
  - `mithril-oxide-sys`: Raw unsafe bindings to the C++ API.
  - `mithril-oxide-sys-codegen`: Automated binding generation (actual type checking and code
    generation).
  - `mithril-oxide-sys-proc`: Automated binding generation (via proc macros).

> Note: The actual code generation needs to be a separate process because Clang's LLVM would
    otherwise collide with Rust's LLVM making the compiler crash.

## Compilation instructions

There'll have to be some environment variables set; in particular:

```bash
# Path to llvm with MLIR.
export MLIR_SYS_160_PREFIX=/usr/lib/llvm-16

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


## Stuff to do

  - Design the Rusty API.

  - Remove hardcoded paths.
  - Improve development experience.
  - Proper error handling for the code generation.
  - Autodetect `Clone`, `Copy` and `Pin` requirements.
  - Implement `Debug` for the generated types.
