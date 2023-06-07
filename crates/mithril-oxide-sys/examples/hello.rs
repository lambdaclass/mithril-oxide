use mithril_oxide_sys::ffi::{Builder, DialectRegistry, MLIRContext, Threading};

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

    //let builder = unsafe {
    //    Builder::new(&mut context)
    //};
    //dbg!(builder);
    unsafe {
        //register_all_dialects(&mut context);
        //println!("all dialects registered")
    };
}
