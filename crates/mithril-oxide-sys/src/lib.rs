#![allow(dead_code)]

use mithril_oxide_sys_proc::codegen;

#[codegen]
pub mod ffi {
    #![codegen(include = "mlir/IR/MLIRContext.h")]
    #![codegen(include = "mlir/InitAllDialects.h")]
    #![codegen(include = "mlir/IR/DialectRegistry.h")]

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

    //#[codegen(cxx_ident = "registerAllDialects")]
    //pub fn register_all_dialects(context: &mut MLIRContext) {}
}
