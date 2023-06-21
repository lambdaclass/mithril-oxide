pub use self::ffi::ModuleOp;
use cxx::UniquePtr;
use std::{fmt, pin::Pin};

use super::Value::ffi::c_void;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinOps.hpp");

        type ModuleOp;
        type Region = crate::IR::Region::Region;
        type Operation = crate::IR::Operation::Operation;

        #[must_use]
        fn getBodyRegion(self: Pin<&mut ModuleOp>) -> Pin<&mut Region>;
        #[must_use]
        pub fn getOperation(self: Pin<&mut ModuleOp>) -> *mut Operation;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinOps.hpp");

        type c_void = crate::c_void;

        unsafe fn ModuleOp_create(loc: *const c_void) -> UniquePtr<ModuleOp>;
        unsafe fn ModuleOp_setSymNameAttr(op: Pin<&mut ModuleOp>, value: *const c_void);
        unsafe fn ModuleOp_setSymVisibilityAttr(op: Pin<&mut ModuleOp>, value: *const c_void);
    }
}

impl ffi::ModuleOp {
    #[must_use]
    pub unsafe fn new(loc: *const c_void) -> UniquePtr<Self> {
        ffi::ModuleOp_create(loc)
    }

    /// value - `StringAttr`
    pub unsafe fn setSymNameAttr(self: Pin<&mut Self>, value: *const c_void) {
        ffi::ModuleOp_setSymNameAttr(self, value);
    }

    /// value - `StringAttr`
    pub unsafe fn setSymVisibilityAttr(self: Pin<&mut Self>, value: *const c_void) {
        ffi::ModuleOp_setSymVisibilityAttr(self, value);
    }
}

impl fmt::Debug for ffi::ModuleOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ModuleOp").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use crate::IR::{Location::UnknownLoc_get, MLIRContext::MLIRContext};

    use super::*;

    #[test]
    fn moduleop_new() {
        let mut context = MLIRContext::new();

        unsafe {
            let loc = UnknownLoc_get(context.pin_mut());
            assert!(!loc.is_null());
            let module_op = ModuleOp::new(loc);
            assert!(!module_op.is_null());
        }
    }
}
