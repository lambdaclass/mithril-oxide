use crate::IR::BuiltinOps::ModuleOp;
use cxx::UniquePtr;
pub use ffi::DataLayout;
use std::fmt;

use crate::ffi::c_void;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Interfaces/DataLayoutInterfaces.hpp");

        type DataLayout;
        type ModuleOp = crate::IR::BuiltinOps::ModuleOp;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Interfaces/DataLayoutInterfaces.hpp");

        type c_void = crate::c_void;

        #[must_use]
        fn DataLayout_new(op: &ModuleOp) -> UniquePtr<DataLayout>;
        #[must_use]
        unsafe fn DataLayout_getTypeSize(layout: &DataLayout, ttype: *const c_void) -> u32;
        #[must_use]
        unsafe fn DataLayout_getTypeABIAlignment(layout: &DataLayout, ttype: *const c_void) -> u32;
        #[must_use]
        unsafe fn DataLayout_getTypePreferredAlignment(
            layout: &DataLayout,
            ttype: *const c_void,
        ) -> u32;
    }
}

impl ffi::DataLayout {
    #[must_use]
    pub fn new(op: &ModuleOp) -> UniquePtr<Self> {
        ffi::DataLayout_new(op)
    }

    #[must_use]
    pub unsafe fn getTypeSize(&self, ttype: *const c_void) -> u32 {
        ffi::DataLayout_getTypeSize(self, ttype)
    }

    #[must_use]
    pub unsafe fn getTypeABIAlignment(&self, ttype: *const c_void) -> u32 {
        ffi::DataLayout_getTypeABIAlignment(self, ttype)
    }

    #[must_use]
    pub unsafe fn getTypePreferredAlignment(&self, ttype: *const c_void) -> u32 {
        ffi::DataLayout_getTypePreferredAlignment(self, ttype)
    }
}

impl fmt::Debug for ffi::DataLayout {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("DataLayout").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use crate::IR::{
        BuiltinTypes::IntegerType_get, Location::UnknownLoc_get, MLIRContext::MLIRContext,
    };

    use super::*;

    #[test]
    fn datalayout() {
        let mut context = MLIRContext::new();

        unsafe {
            let loc = UnknownLoc_get(context.pin_mut());
            assert!(!loc.is_null());
            let module_op = ModuleOp::new(loc);
            assert!(!module_op.is_null());
            let datalayout = DataLayout::new(&module_op);

            let int_type = IntegerType_get(context.pin_mut(), 8, false, false);
            let size = datalayout.getTypeSize(int_type);
            assert_eq!(size, 1);

            let int_type = IntegerType_get(context.pin_mut(), 16, false, false);
            let size = datalayout.getTypeSize(int_type);
            assert_eq!(size, 2);
        }
    }
}
