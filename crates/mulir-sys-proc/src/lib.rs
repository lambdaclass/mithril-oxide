use proc_macro as pm;
use proc_macro2 as pm2;
use syn::ItemMod;
use crate::codegen::codegen_cpp;

mod codegen;
mod parse;
mod request;

#[proc_macro_attribute]
pub fn codegen(attr: pm::TokenStream, input: pm::TokenStream) -> pm::TokenStream {
    codegen_impl(attr.into(), input.into()).into()
}

fn codegen_impl(attr: pm2::TokenStream, input: pm2::TokenStream) -> pm2::TokenStream {
    assert!(attr.is_empty());

    // TODO: Parse macro input into bindings to generate.
    let item_mod = syn::parse2::<ItemMod>(input).unwrap();
    let request = parse::parse_macro_input(item_mod).unwrap();

    // TODO: Generate C++ source file.
    let cpp_source = codegen_cpp(&request);
    dbg!(cpp_source);
    // TODO: Parse C++ source file.

    // TODO: Generate Rust bindings.
    // TODO: Build C++ source file.

    // let ffi_mod = syn::parse2::<syn::ItemMod>(input).unwrap();
    // codegen::codegen_mod(&ffi_mod);

    todo!()
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

                    #[codegen(cxx_path = "mlir::MLIRContext", kind = "opaque-sized")]
                    pub struct MlirContext;

                    impl MlirContext {
                        #[codegen(constructor)]
                        pub fn new() -> Self;

                        pub fn enableSomething(&mut self);
                    }
                }
            },
        );
    }
}
