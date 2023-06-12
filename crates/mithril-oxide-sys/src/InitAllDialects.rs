pub use self::ffi::registerAllDialects;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/InitAllDialects.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;

        fn registerAllDialects(context: Pin<&mut MLIRContext>);
    }
}
