use proc_macro2::TokenStream;
use std::io::{stdin, stdout, Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read input code.
    let mut data = String::new();
    stdin().read_to_string(&mut data)?;

    // Process code.
    let stream: TokenStream = data.parse()?;
    let stream = mithril_oxide_sys_codegen::codegen(stream)?;

    // Write output code.
    let data = stream.to_string();
    stdout().write_all(data.as_bytes())?;

    Ok(())
}
