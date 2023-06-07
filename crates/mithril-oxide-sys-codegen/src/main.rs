use proc_macro2::TokenStream;
use std::{
    env::args,
    io::{stdin, stdout, Read, Write},
    path::Path,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read input code.
    let mut data = String::new();
    stdin().read_to_string(&mut data)?;

    // Process code.
    let stream: TokenStream = data.parse()?;
    let stream = mithril_oxide_sys_codegen::codegen(
        Path::new(
            &args()
                .nth(1)
                .expect("Auxiliary library target path required."),
        ),
        stream,
    )?;

    // Write output code.
    let data = stream.to_string();
    stdout().write_all(data.as_bytes())?;

    Ok(())
}
