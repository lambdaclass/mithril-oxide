use mithril_oxide_sys::ffi::{registerAllDialects, DialectRegistry, MLIRContext, Threading};

fn main() {
    let mut context = unsafe { MLIRContext::new(Threading::DISABLED) };
    println!("MLIR context is multithreaded? {}", unsafe {
        context.isMultithreadingEnabled()
    });

    let mut context = unsafe { MLIRContext::new(Threading::ENABLED) };
    println!("MLIR context is multithreaded? {}", unsafe {
        context.isMultithreadingEnabled()
    });

    let registry = unsafe { DialectRegistry::new() };
    unsafe {
        registerAllDialects(&mut context);
        //println!("all dialects registered")
    };
}
