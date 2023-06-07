use mithril_oxide_sys_codegen::codegen;
use quote::quote;

#[test]
fn mlir() {
    println!(
        "{}",
        codegen(quote! {
            #[codegen]
            extern {
                include!("mlir/InitAllDialects.h");
                include!("mlir/IR/MLIRContext.h");

                #[codegen(cxx_path = "mlir::MLIRContext::Threading")]
                pub enum Threading {
                    DISABLED,
                    ENABLED,
                }

                #[codegen(cxx_path = "mlir::MLIRContext", kind = "opaque-sized")]
                pub struct MLIRContext;

                #[codegen(cxx_path = "mlir::registerAllDialects")]
                pub fn registerAllDialects(context: &mut MLIRContext);

                impl MLIRContext {
                    #[codegen(cxx_path = "MLIRContext")]
                    pub fn new(threading: Threading) -> Self;

                    // pub fn isMultithreadingEnabled(&mut self) -> bool;
                }

                // #[codegen(cxx_path = "mlir::DialectRegistry", kind = "opaque-sized")]
                // pub struct DialectRegistry;
                // impl DialectRegistry {
                //     #[codegen(constructor)]
                //     pub fn new() -> Self;
                // }
                // // https://mlir.llvm.org/doxygen/classmlir_1_1Type.html
                // #[codegen(cxx_path = "mlir::Type", kind = "opaque-sized")]
                // pub struct Type;
                // impl Type {
                //     // types are usually made from op builder
                //     pub fn isIndex(&self) -> bool;
                //     pub fn isFloat8E5M2(&self) -> bool;
                //     pub fn isFloat8E4M3FN(&self) -> bool;
                //     pub fn isBF16(&self) -> bool;
                //     pub fn isF16(&self) -> bool;
                //     pub fn isF32(&self) -> bool;
                //     pub fn isF64(&self) -> bool;
                //     pub fn isF80(&self) -> bool;
                //     pub fn isF128(&self) -> bool;
                //     pub fn isInteger(&self, width: u32) -> bool;
                //     pub fn isSignlessInteger(&self) -> bool;
                //     // pub fn isSignlessInteger(&self, width: u32) -> bool;
                //     pub fn isUnsignedInteger(&self) -> bool;
                //     // pub fn isUnsignedInteger(&self, width: u32) -> bool;
                //     pub fn getIntOrFloatBitWidth() -> u32;
                //     pub fn isSignlessIntOrIndex(&self) -> bool;
                //     pub fn isSignlessIntOrIndexOrFloat(&self) -> bool;
                //     pub fn isSignlessIntOrFloat(&self) -> bool;
                //     pub fn isIntOrIndex(&self) -> bool;
                //     pub fn isIntOrFloat(&self) -> bool;
                //     pub fn isIntOrIndexOrFloat(&self) -> bool;
                //     pub fn dump(&self);
                // }

                // #[codegen(cxx_path = "mlir::Builder", kind = "opaque-sized")]
                // pub struct Builder;

                // impl Builder {
                //     //#[codegen(constructor)]
                //     //pub fn new(context: *mut MLIRContext) -> Self;
                // }
            }
        })
        .unwrap()
    );
}
