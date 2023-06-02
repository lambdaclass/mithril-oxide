use mithril_oxide_sys::ffi::{MlirContext, Threading};

fn main() {
    let mut context = unsafe { MlirContext::new(Threading::DISABLED) };
    println!("MLIR context is multithreaded? {}", unsafe {
        context.isMultithreadingEnabled()
    });

    let mut context = unsafe { MlirContext::new(Threading::ENABLED) };
    println!("MLIR context is multithreaded? {}", unsafe {
        context.isMultithreadingEnabled()
    });
}
