pub use mithril_oxide_sys::*;

#[test]
fn context_new() {
    unsafe {
        let context = MLIRContext::new(Threading::DISABLED);
        context.del();
    }
}

#[test]
fn context_multithreaded() {
    unsafe {
        let mut context = MLIRContext::new(Threading::DISABLED);
        assert!(!context.isMultithreadingEnabled());
        context.del();
    }

    unsafe {
        let mut context = MLIRContext::new(Threading::ENABLED);
        assert!(context.isMultithreadingEnabled());
        context.del();
    }
}

#[test]
fn context_register_load_dialects() {
    unsafe {
        let mut context = MLIRContext::new(Threading::DISABLED);
        registerAllDialects(&mut context);
        context.loadAllAvailableDialects();
        context.del();
    }
}
