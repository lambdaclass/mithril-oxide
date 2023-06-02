use mulir_sys_proc::codegen;

#[codegen]
mod ffi {
    #![codegen(include = "mlir/IR/MLIRContext.h")]

    #[codegen(cxx_type = "mlir::MLIRContext", kind = "fat")]
    pub struct MlirContext;

    impl MlirContext {
        #[codegen(constructor)]
        pub fn new();
    }
}
