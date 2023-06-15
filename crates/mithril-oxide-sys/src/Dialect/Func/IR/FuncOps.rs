use self::ffi::c_void;
pub use self::ffi::{FuncOp, ReturnOp};
use crate::IR::{Attributes::NamedAttribute, Location::Location};
use cxx::UniquePtr;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {

    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Func/IR/FuncOps.hpp");

        #[namespace = "mlir::func"]
        type FuncOp;
        #[namespace = "mlir::func"]
        type ReturnOp;
        type Location = crate::IR::Location::Location;
        type NamedAttribute = crate::IR::Attributes::NamedAttribute;
        type Operation = crate::IR::Operation::Operation;

        #[must_use]
        pub fn getOperation(self: Pin<&mut FuncOp>) -> *mut Operation;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Func/IR/FuncOps.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        unsafe fn FuncOp_create(
            context: &Location,
            name: &str,
            func_type: *const c_void,
            attrs: &[*const NamedAttribute],
            // DictionaryAttr
            argAttrs: &[*const c_void],
        ) -> UniquePtr<FuncOp>;

        unsafe fn ReturnOp_create(
            context: &Location,
            operands: &[*const c_void],
        ) -> UniquePtr<ReturnOp>;
    }
}

impl ffi::FuncOp {
    #[must_use]
    /// argAttrs is a array of DictionaryAttr
    pub unsafe fn new<'a>(
        context: &Location,
        name: &str,
        function_type: *const c_void,
        attrs: impl IntoIterator<Item = &'a NamedAttribute>,
        argAttrs: impl IntoIterator<Item = *const c_void>,
    ) -> UniquePtr<Self> {
        let attrs_vec = attrs.into_iter().map(|x| x as *const _).collect::<Vec<_>>();
        let argAttrs_vec = argAttrs.into_iter().collect::<Vec<_>>();

        ffi::FuncOp_create(context, name, function_type, &attrs_vec, &argAttrs_vec)
    }
}

impl ffi::ReturnOp {
    #[must_use]
    pub unsafe fn new(
        loc: &Location,
        attrs: impl IntoIterator<Item = *const c_void>,
    ) -> UniquePtr<Self> {
        let operands = attrs.into_iter().collect::<Vec<_>>();

        ffi::ReturnOp_create(loc, &operands)
    }
}

impl fmt::Debug for ffi::FuncOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FuncOp").finish_non_exhaustive()
    }
}
