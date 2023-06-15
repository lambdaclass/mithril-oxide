use self::ffi::Value;
pub use self::ffi::{FuncOp, ReturnOp};
use crate::IR::{
    Attributes::NamedAttribute, BuiltinAttributes::DictionaryAttr, BuiltinTypes::FunctionType,
    Location::Location,
};
use cxx::UniquePtr;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {

    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Func/IR/FuncOps.hpp");

        type DictionaryAttr = crate::IR::BuiltinAttributes::DictionaryAttr;
        #[namespace = "mlir::func"]
        type FuncOp;
        #[namespace = "mlir::func"]
        type ReturnOp;
        type FunctionType = crate::IR::BuiltinTypes::FunctionType;
        type Location = crate::IR::Location::Location;
        type NamedAttribute = crate::IR::Attributes::NamedAttribute;
        type Operation = crate::IR::Operation::Operation;
        type Value = crate::IR::Value::Value;

        #[must_use]
        pub fn getOperation(self: Pin<&mut FuncOp>) -> *mut Operation;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Func/IR/FuncOps.hpp");

        fn FuncOp_create(
            context: &Location,
            name: &str,
            type_: &FunctionType,
            attrs: &[*const NamedAttribute],
            argAttrs: &[*const DictionaryAttr],
        ) -> UniquePtr<FuncOp>;

        fn ReturnOp_create(context: &Location, operands: &[*const Value]) -> UniquePtr<ReturnOp>;
    }
}

impl ffi::FuncOp {
    #[must_use]
    pub fn new<'a>(
        context: &Location,
        name: &str,
        r#type: &FunctionType,
        attrs: impl IntoIterator<Item = &'a NamedAttribute>,
        argAttrs: impl IntoIterator<Item = &'a DictionaryAttr>,
    ) -> UniquePtr<Self> {
        let attrs_vec = attrs.into_iter().map(|x| x as *const _).collect::<Vec<_>>();
        let argAttrs_vec = argAttrs
            .into_iter()
            .map(|x| x as *const _)
            .collect::<Vec<_>>();

        ffi::FuncOp_create(context, name, r#type, &attrs_vec, &argAttrs_vec)
    }
}

impl ffi::ReturnOp {
    #[must_use]
    pub fn new<'a>(loc: &Location, attrs: impl IntoIterator<Item = &'a Value>) -> UniquePtr<Self> {
        let operands = attrs.into_iter().map(|x| x as *const _).collect::<Vec<_>>();

        ffi::ReturnOp_create(loc, &operands)
    }
}

impl fmt::Debug for ffi::FuncOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FuncOp").finish_non_exhaustive()
    }
}
