pub use self::ffi::FuncOp;
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
        type FunctionType = crate::IR::BuiltinTypes::FunctionType;
        type Location = crate::IR::Location::Location;
        type NamedAttribute = crate::IR::Attributes::NamedAttribute;
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

impl fmt::Debug for ffi::FuncOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FuncOp").finish_non_exhaustive()
    }
}
