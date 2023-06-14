use mithril_oxide_sys::{
    registerAllDialects, test_ptr_cpp, MLIRContext, ModuleOp_create, Threading, UnknownLoc,
    UnknownLoc_getContext, UnknownLoc_to_Location,
};

fn main() {
    unsafe {
        let mut context = MLIRContext::new(Threading::DISABLED);
        println!(
            "MLIR context is multithreaded? {}",
            context.isMultithreadingEnabled()
        );

        registerAllDialects(&mut context);
        context.loadAllAvailableDialects();

        let mut loc = UnknownLoc::get(&mut context as *mut _);

        println!(
            "[Rust] ptr(context)    = {:?}",
            &mut context as *mut MLIRContext
        );
        UnknownLoc_getContext(&mut loc);

        test_ptr_cpp();

        let mut loc = UnknownLoc_to_Location(&mut loc);
        let _module_op = ModuleOp_create(&mut loc);

        context.del();
    }
}
