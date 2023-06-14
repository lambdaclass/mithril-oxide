use self::ffi::*;
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
        pub fn DenseIntElementsAttr_to_Attribute(
            attr: &DenseIntElementsAttr,
        ) -> UniquePtr<Attribute>;
        pub fn DenseFPElementsAttr_to_Attribute(attr: &DenseFPElementsAttr)
            -> UniquePtr<Attribute>;
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

macro_rules! impl_attribute_conversion {
    ($type_name:ident, $func_name:ident) => {
        impl From<&crate::IR::BuiltinAttributes::$type_name>
            for UniquePtr<crate::IR::Attributes::Attribute>
        {
            fn from(val: &crate::IR::BuiltinAttributes::$type_name) -> Self {
                crate::IR::BuiltinAttributes::$func_name(val)
            }
        }
    };
}

impl_attribute_conversion!(StringAttr, StringAttr_to_Attribute);
impl_attribute_conversion!(FloatAttr, FloatAttr_to_Attribute);
impl_attribute_conversion!(IntegerAttr, IntegerAttr_to_Attribute);
impl_attribute_conversion!(DenseElementsAttr, DenseElementsAttr_to_Attribute);
impl_attribute_conversion!(DenseIntElementsAttr, DenseIntElementsAttr_to_Attribute);
impl_attribute_conversion!(DenseFPElementsAttr, DenseFPElementsAttr_to_Attribute);
impl_attribute_conversion!(BoolAttr, BoolAttr_to_Attribute);
impl_attribute_conversion!(FlatSymbolRefAttr, FlatSymbolRefAttr_to_Attribute);
impl_attribute_conversion!(DictionaryAttr, DictionaryAttr_to_Attribute);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_attr_new() {
        let mut context = MLIRContext::new();
        let string_attr = StringAttr::new(context.pin_mut(), "hello_world");
        let attr: UniquePtr<Attribute> = (&*string_attr).into();
        assert_eq!("\"hello_world\"", attr.print().as_str());
    }
}
