use mithril_oxide_sys::{registerAllDialects, MLIRContext, Threading};

fn main() {
    unsafe {
        let mut context = MLIRContext::new(Threading::DISABLED);
        println!(
            "MLIR context is multithreaded? {}",
            context.isMultithreadingEnabled()
        );

        registerAllDialects(&mut context);
        context.loadAllAvailableDialects();

        context.del();
    }
}
