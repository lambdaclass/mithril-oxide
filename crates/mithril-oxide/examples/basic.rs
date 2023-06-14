use mithril_oxide::prelude::*;

fn main() {
    let mut context = Context::new();
    context.register_all_dialects();
    context.load_all_available_dialects();

    let mut module = ops::builtin::ModuleOp::builder()
        .location(loc::UnknownLoc::new(&context))
        .build(&context);

    let mut body = module.body_mut().emplace_block();
    let func_op = body.push(
        ops::func::FuncOp::builder()
            .location(loc::UnknownLoc::new(&context))
            .sym_name("main")
            .function_type(todo!())
            .sym_visibility("public"),
    );

    let main_block = func_op.body_mut().emplace_block();
    mithril_oxide::mlir_asm! { main_block =>
        ; t0 = "arith.constant"() { value = 0 } : () -> i8
        ; "func.return"(t0) : i8 -> ()
    }

    println!("{module}");
}
