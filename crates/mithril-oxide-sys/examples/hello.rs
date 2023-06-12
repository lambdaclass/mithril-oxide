use cxx::UniquePtr;
use mithril_oxide_sys::{
    InitAllDialects::registerAllDialects,
    IR::{
        Location::{Location, UnknownLoc},
        MLIRContext::MLIRContext, BuiltinOps::ModuleOp,
    },
};

fn main() {
    let mut context = MLIRContext::new();
    registerAllDialects(context.pin_mut());

    let loc = UnknownLoc::get(context.pin_mut());
    let loc: UniquePtr<Location> = (&*loc).into();

    let module_op = ModuleOp::new(&loc);

    dbg!(context, loc, module_op);
}
