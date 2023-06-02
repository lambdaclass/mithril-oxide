use mulir_sys::ffi::Threading;

fn main() {
    let mut context = unsafe { mulir_sys::ffi::MlirContext::new(Threading::DISABLED) };
    println!("MLIR context is multithreaded? {}", unsafe {
        context.isMultithreadingEnabled()
    });

    let mut context = unsafe { mulir_sys::ffi::MlirContext::new(Threading::ENABLED) };
    println!("MLIR context is multithreaded? {}", unsafe {
        context.isMultithreadingEnabled()
    });
}
