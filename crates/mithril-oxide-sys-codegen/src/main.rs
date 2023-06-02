#![allow(dead_code)]
#![allow(unused)]

use analysis::{analyze_cpp, load_cpp};
use clang::{Clang, Index};
use codegen::{codegen_cpp, codegen_rust};
use proc_macro2::TokenStream;
use quote::quote;
use std::{
    io::{stdin, stdout, Read, Write},
    str::FromStr,
};
use syn::ItemMod;

mod analysis;
mod codegen;
mod parse;
mod request;

fn main() {
    let mut data = String::new();
    stdin().read_to_string(&mut data).unwrap();

    let stream = TokenStream::from_str(&data).unwrap();
    let stream = codegen_impl(stream);

    stdout().write_all(stream.to_string().as_bytes()).unwrap();
}

fn codegen_impl(input: TokenStream) -> TokenStream {
    let clang = Clang::new().unwrap();
    let index = Index::new(&clang, true, true);

    // Parse macro input into bindings to generate.
    let item_mod = syn::parse2::<ItemMod>(input).unwrap();
    let mod_name = item_mod.ident.clone();
    let mod_vis = item_mod.vis.clone();
    let request = parse::parse_macro_input(item_mod).unwrap();

    // Generate C++ source file.
    let cpp_source = codegen_cpp(&request);
    let translation_unit = load_cpp(&index, &cpp_source);

    // Parse C++ source file.
    let mappings = analyze_cpp(&translation_unit, &request);

    // Generate Rust bindings.
    let stream = codegen_rust(&mappings);

    quote! {
        #mod_vis mod #mod_name {
            #stream
        }
    }
}
