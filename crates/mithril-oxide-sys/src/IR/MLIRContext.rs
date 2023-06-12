pub use self::ffi::MLIRContext;
use cxx::UniquePtr;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/MLIRContext.hpp");

        type MLIRContext;

        pub fn loadAllAvailableDialects(self: Pin<&mut MLIRContext>);
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/MLIRContext.hpp");

        fn MLIRContext_new() -> UniquePtr<MLIRContext>;
    }
}

impl ffi::MLIRContext {
    pub fn new() -> UniquePtr<Self> {
        ffi::MLIRContext_new()
    }
}

impl fmt::Debug for ffi::MLIRContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MLIRContext").finish_non_exhaustive()
    }
}
