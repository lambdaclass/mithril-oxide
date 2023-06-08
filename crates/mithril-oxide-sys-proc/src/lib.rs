#![deny(clippy::pedantic)]
#![deny(warnings)]
#![feature(proc_macro_diagnostic)]

use pm2::TokenStream;
use proc_macro as pm;
use proc_macro2 as pm2;
use std::{
    env::var,
    io::{Read, Write},
    process::{Command, Stdio},
    str::FromStr,
};

#[proc_macro_attribute]
pub fn codegen(attr: pm::TokenStream, input: pm::TokenStream) -> pm::TokenStream {
    codegen_impl(attr.into(), input.into()).into()
}

#[allow(clippy::needless_pass_by_value)]
fn codegen_impl(attr: pm2::TokenStream, input: pm2::TokenStream) -> pm2::TokenStream {
    if !attr.is_empty() {
        let mut iter = attr.into_iter().map(|x| x.span());
        let first = iter.next().unwrap();
        iter.fold(first, |acc, span| acc.join(span).unwrap())
            .unwrap()
            .error("Unsupported attribute location.")
            .emit();

        return TokenStream::new();
    }

    let codegen_path = match env!("PROFILE") {
        "debug" => format!(
            "{}/../../target/debug/mithril-oxide-sys-codegen",
            env!("CARGO_MANIFEST_DIR")
        ),
        "release" => format!(
            "{}/../../target/release/mithril-oxide-sys-codegen",
            env!("CARGO_MANIFEST_DIR")
        ),
        _ => panic!("Unsupported profile name."),
    };

    let mut process = Command::new(codegen_path)
        .arg(format!("{}/libauxlib.a", var("OUT_DIR").unwrap()))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
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
