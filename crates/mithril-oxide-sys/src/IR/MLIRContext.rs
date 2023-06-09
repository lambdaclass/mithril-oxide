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

        #[must_use]
        pub fn allowsUnregisteredDialects(self: Pin<&mut MLIRContext>) -> bool;
        pub fn allowUnregisteredDialects(self: Pin<&mut MLIRContext>, allow: bool);

        pub fn enableMultithreading(self: Pin<&mut MLIRContext>, enable: bool);
        #[must_use]
        pub fn isMultithreadingEnabled(self: Pin<&mut MLIRContext>) -> bool;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/MLIRContext.hpp");

        fn MLIRContext_new() -> UniquePtr<MLIRContext>;
    }
}

impl ffi::MLIRContext {
    #[must_use]
    pub fn new() -> UniquePtr<Self> {
        ffi::MLIRContext_new()
    }
}

impl fmt::Debug for ffi::MLIRContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MLIRContext").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InitAllDialects::registerAllDialects;

    #[test]
    fn context_new() {
        let mut context = MLIRContext::new();
        registerAllDialects(context.pin_mut());
    }
}
