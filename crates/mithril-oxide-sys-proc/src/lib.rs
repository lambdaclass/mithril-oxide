use proc_macro as pm;
use proc_macro2 as pm2;
use std::{
    io::{Read, Write},
    process::{Command, Stdio},
    str::FromStr,
};

#[proc_macro_attribute]
pub fn codegen(attr: pm::TokenStream, input: pm::TokenStream) -> pm::TokenStream {
    codegen_impl(attr.into(), input.into()).into()
}

fn codegen_impl(attr: pm2::TokenStream, input: pm2::TokenStream) -> pm2::TokenStream {
    assert!(attr.is_empty());

    let mut process = Command::new("/home/esteve/Documents/LambdaClass/mulir/target/debug/mulir-sys-codegen")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    process
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input.to_string().as_bytes())
        .unwrap();
    process.wait().unwrap();

    let mut output = String::new();
    process.stdout.unwrap().read_to_string(&mut output).unwrap();

    pm2::TokenStream::from_str(&output).unwrap()
}
