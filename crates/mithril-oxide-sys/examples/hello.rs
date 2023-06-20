use mithril_oxide_sys::{
    InitAllDialects::registerAllDialects,
    IR::{BuiltinOps::ModuleOp, Location::UnknownLoc_get, MLIRContext::MLIRContext},
};

fn main() {
    let mut context = MLIRContext::new();
    registerAllDialects(context.pin_mut());

    unsafe {
        let loc = UnknownLoc_get(context.pin_mut());

        let module_op = ModuleOp::new(loc);

        dbg!(context, loc, module_op);
    }
}
