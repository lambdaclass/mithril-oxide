#![allow(dead_code)]

use mithril_oxide_sys_proc::codegen;

#[codegen]
pub mod ffi {
    #![codegen(include = "mlir/IR/MLIRContext.h")]
    #![codegen(include = "mlir/InitAllDialects.h")]
    #![codegen(include = "mlir/IR/DialectRegistry.h")]
    #![codegen(include = "mlir/IR/Types.h")]

    #[codegen(cxx_path = "mlir::MLIRContext::Threading")]
    pub enum Threading {
        DISABLED,
        ENABLED,
    }

    #[codegen(cxx_path = "mlir::MLIRContext", kind = "opaque-sized")]
    pub struct MLIRContext;

    impl MLIRContext {
        #[codegen(constructor)]
        pub fn new(threading: Threading) -> Self;

        pub fn isMultithreadingEnabled(&mut self) -> bool;
    }

    #[codegen(cxx_path = "mlir::DialectRegistry", kind = "opaque-sized")]
    pub struct DialectRegistry;

    impl DialectRegistry {
        #[codegen(constructor)]
        pub fn new() -> Self;
    }

    // https://mlir.llvm.org/doxygen/classmlir_1_1Type.html
    #[codegen(cxx_path = "mlir::Type", kind = "opaque-sized")]
    pub struct Type;

    impl Type {
        // types are usually made from op builder

        pub fn isIndex(&self) -> bool;
        pub fn isFloat8E5M2(&self) -> bool;
        pub fn isFloat8E4M3FN(&self) -> bool;
        pub fn isBF16(&self) -> bool;
        pub fn isF16(&self) -> bool;
        pub fn isF32(&self) -> bool;
        pub fn isF64(&self) -> bool;
        pub fn isF80(&self) -> bool;
        pub fn isF128(&self) -> bool;

        pub fn isInteger(&self, width: u32) -> bool;
    }

    //#[codegen(cxx_ident = "registerAllDialects")]
    //pub fn register_all_dialects(context: &mut MLIRContext) {}
}
