use mithril_oxide_sys::ffi::{DialectRegistry, MLIRContext, Threading};

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
        //register_all_dialects(&mut context);
        //println!("all dialects registered")
    };
}
