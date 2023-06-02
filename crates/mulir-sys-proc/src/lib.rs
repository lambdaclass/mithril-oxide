use crate::{
    analysis::{analyze_cpp, load_cpp},
    codegen::{codegen_cpp, codegen_rust},
};
use clang::{Clang, Index};
use proc_macro as pm;
use proc_macro2 as pm2;
use syn::ItemMod;

mod analysis;
mod codegen;
mod parse;
mod request;

#[proc_macro_attribute]
pub fn codegen(attr: pm::TokenStream, input: pm::TokenStream) -> pm::TokenStream {
    codegen_impl(attr.into(), input.into()).into()
}

fn codegen_impl(attr: pm2::TokenStream, input: pm2::TokenStream) -> pm2::TokenStream {
    assert!(attr.is_empty());

    let clang = Clang::new().unwrap();
    let index = Index::new(&clang, true, true);

    // TODO: Parse macro input into bindings to generate.
    let item_mod = syn::parse2::<ItemMod>(input).unwrap();
    let request = parse::parse_macro_input(item_mod).unwrap();

    // TODO: Generate C++ source file.
    let cpp_source = codegen_cpp(&request);
    let translation_unit = load_cpp(&index, &cpp_source);

    // TODO: Parse C++ source file.
    let mappings = analyze_cpp(&translation_unit, &request);

    // TODO: Generate Rust bindings.
    codegen_rust(&mappings)
}

#[cfg(test)]
mod test {
    use crate::codegen_impl;
    use quote::quote;

    #[test]
    fn test() {
        codegen_impl(
            quote! {},
            quote! {
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
            },
        );
    }
}
