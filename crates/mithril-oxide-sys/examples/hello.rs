use cxx::UniquePtr;
use mithril_oxide_sys::{
    InitAllDialects::registerAllDialects,
    IR::{
        BuiltinOps::ModuleOp,
        Location::{Location, UnknownLoc},
        MLIRContext::MLIRContext,
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
