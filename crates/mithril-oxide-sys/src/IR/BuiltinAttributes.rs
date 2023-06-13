pub use self::ffi::StringAttr;
use crate::IR::MLIRContext::MLIRContext;
use cxx::UniquePtr;
use std::{fmt, pin::Pin};

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinAttributes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
        type StringAttr;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinAttributes.hpp");

        fn StringAttr_get(context: Pin<&mut MLIRContext>, value: &str) -> UniquePtr<StringAttr>;
    }
}

impl ffi::StringAttr {
    pub fn new(context: Pin<&mut MLIRContext>, value: &str) -> UniquePtr<Self> {
        ffi::StringAttr_get(context, value)
    }
}

impl fmt::Debug for ffi::StringAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("StringAttr").finish_non_exhaustive()
    }
}
