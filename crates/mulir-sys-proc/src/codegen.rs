use crate::request::RequestMod;

pub fn codegen_cpp(request: &RequestMod) -> String {
    let mut output = String::new();

    output.push_str("#include <type_traits>\n\n");
    for include in &request.includes {
        output.push_str(&format!("#include <{include}>\n"));
    }
    output.push_str("\n\n");

    output
}
