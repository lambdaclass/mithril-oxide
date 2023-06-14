pub use self::ffi::{DictionaryAttr, StringAttr};
use crate::IR::MLIRContext::MLIRContext;
use cxx::UniquePtr;
use std::{fmt, pin::Pin};

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinAttributes.hpp");

        type DictionaryAttr;
        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
        type Attribute = crate::IR::Attributes::Attribute;
        type StringAttr;
        type FloatAttr;
        type IntegerAttr;
        type DenseElementsAttr;
        type DenseIntElementsAttr;
        type DenseFPElementsAttr;
        type BoolAttr;
        type FlatSymbolRefAttr;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinAttributes.hpp");

        fn StringAttr_get(context: Pin<&mut MLIRContext>, value: &str) -> UniquePtr<StringAttr>;

        pub fn StringAttr_to_Attribute(attr: &StringAttr) -> UniquePtr<Attribute>;
        pub fn FloatAttr_to_Attribute(attr: &FloatAttr) -> UniquePtr<Attribute>;
        pub fn IntegerAttr_to_Attribute(attr: &IntegerAttr) -> UniquePtr<Attribute>;
        pub fn DenseElementsAttr_to_Attribute(attr: &DenseElementsAttr) -> UniquePtr<Attribute>;
        pub fn DenseIntElementsAttr_to_Attribute(attr: &DenseIntElementsAttr) -> UniquePtr<Attribute>;
        pub fn DenseFPElementsAttr_to_Attribute(attr: &DenseFPElementsAttr) -> UniquePtr<Attribute>;
        pub fn BoolAttr_to_Attribute(attr: &BoolAttr) -> UniquePtr<Attribute>;
        pub fn FlatSymbolRefAttr_to_Attribute(attr: &FlatSymbolRefAttr) -> UniquePtr<Attribute>;
        pub fn DictionaryAttr_to_Attribute(attr: &DictionaryAttr) -> UniquePtr<Attribute>;
    }
}

impl ffi::StringAttr {
    #[must_use]
    pub fn new(context: Pin<&mut MLIRContext>, value: &str) -> UniquePtr<Self> {
        ffi::StringAttr_get(context, value)
    }
}

impl fmt::Debug for ffi::StringAttr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("StringAttr").finish_non_exhaustive()
    }
}
