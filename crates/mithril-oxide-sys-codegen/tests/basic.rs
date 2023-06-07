use mithril_oxide_sys_codegen::codegen;
use quote::quote;
use std::path::Path;

#[test]
fn basic() {
    codegen(
        Path::new(""),
        quote! {
            #[codegen]
            extern {
                #![include = "mlir/IR/MLIRContext.h"]

                #[codegen(cxx_path = "mlir::MLIRContext", kind = "opaque-sized")]
                pub struct MLIRContext;

                impl MLIRContext {
                    #[codegen(constructor)]
                    fn new() -> Self;
                }

                #[codegen(cxx_path = "mlir::MLIRContext::ThreadingEnabled")]
                enum ThreadingEnabled {
                    DISABLED,
                    ENABLED,
                }

                #[codegen(cxx_path = "registerAllDialects")]
                fn registerAllDialects_DialectRegistry(context: &mut DialectRegistry);
                #[codegen(cxx_path = "registerAllDialects")]
                fn registerAllDialects_MLIRContext(context: &mut MLIRContext);
            }
        },
    )
    .unwrap();
}
