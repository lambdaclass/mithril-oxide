pub use self::ffi::ModuleOp;
use crate::IR::{BuiltinAttributes::StringAttr, Location::Location};
use cxx::UniquePtr;
use std::{fmt, pin::Pin};

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinOps.hpp");

        type Location = crate::IR::Location::Location;
        type ModuleOp;
        type Region = crate::IR::Region::Region;
        type StringAttr = crate::IR::BuiltinAttributes::StringAttr;

        fn getBodyRegion(self: Pin<&mut ModuleOp>) -> Pin<&mut Region>;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinOps.hpp");

        fn ModuleOp_create(context: &Location) -> UniquePtr<ModuleOp>;
        fn ModuleOp_setSymNameAttr(op: Pin<&mut ModuleOp>, value: &StringAttr);
        fn ModuleOp_setSymVisibilityAttr(op: Pin<&mut ModuleOp>, value: &StringAttr);
    }
}

impl ffi::ModuleOp {
    pub fn new(context: &Location) -> UniquePtr<Self> {
        ffi::ModuleOp_create(context)
    }

    pub fn setSymNameAttr(self: Pin<&mut Self>, value: StringAttr) {
        ffi::ModuleOp_setSymNameAttr(self, &value);
    }

    pub fn setSymVisibilityAttr(self: Pin<&mut Self>, value: StringAttr) {
        ffi::ModuleOp_setSymVisibilityAttr(self, &value);
    }
}

impl fmt::Debug for ffi::ModuleOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ModuleOp").finish_non_exhaustive()
    }
}
