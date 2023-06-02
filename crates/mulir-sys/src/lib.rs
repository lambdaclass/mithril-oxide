use mulir_sys_proc::codegen;

#[codegen]
mod ffi {
    #![codegen(include = "mlir/IR/MLIRContext.h")]

    #[codegen(cxx_path = "mlir::MLIRContext::Threading")]
    pub enum Threading {
        DISABLED,
        ENABLED,
    }

    #[codegen(cxx_path = "mlir::MLIRContext", kind = "opaque-sized")]
    pub struct MlirContext;

    impl MlirContext {
        #[codegen(constructor)]
        pub fn new() -> Self;

        pub fn isMultithreadingEnabled(&mut self) -> bool;
    }
}
