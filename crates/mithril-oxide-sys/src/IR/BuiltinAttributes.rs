pub use self::ffi::StringAttr;
use crate::IR::MLIRContext::MLIRContext;
use cxx::UniquePtr;
use std::{fmt, pin::Pin, ptr::null};

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

        unsafe fn StringAttr_get(
            context: Pin<&mut MLIRContext>,
            value: *const &str,
        ) -> UniquePtr<StringAttr>;
    }
}

impl ffi::StringAttr {
    pub fn empty(context: Pin<&mut MLIRContext>) -> UniquePtr<Self> {
        unsafe { ffi::StringAttr_get(context, null()) }
    }

    pub fn with_value(context: Pin<&mut MLIRContext>, value: &str) -> UniquePtr<Self> {
        unsafe { ffi::StringAttr_get(context, &value as *const &str) }
    }
}

impl fmt::Debug for ffi::StringAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("StringAttr").finish_non_exhaustive()
    }
}
