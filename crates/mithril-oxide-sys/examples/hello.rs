use mithril_oxide_sys::ffi::{register_all_dialects, MLIRContext, Threading};

fn main() {
    let mut context = unsafe { MLIRContext::new(Threading::DISABLED) };
    println!("MLIR context is multithreaded? {}", unsafe {
        context.isMultithreadingEnabled()
    });

    let mut context = unsafe { MLIRContext::new(Threading::ENABLED) };
    println!("MLIR context is multithreaded? {}", unsafe {
        context.isMultithreadingEnabled()
    });
    unsafe {
        register_all_dialects(&mut context);
        println!("all dialects registered")
    };
}
