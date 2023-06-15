pub use self::ffi::{
    BoolAttr_get, DenseElementsAttr_get, DictionaryAttr_get, IntegerAttr_get, StringAttr_get,
};

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinAttributes.hpp");

        type NamedAttribute = crate::IR::Attributes::NamedAttribute;
        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinAttributes.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        #[must_use]
        fn StringAttr_get(context: Pin<&mut MLIRContext>, value: &str) -> *const c_void;
        #[must_use]
        fn IntegerAttr_get(context: Pin<&mut MLIRContext>, value: &str) -> *const c_void;
        #[must_use]
        fn BoolAttr_get(context: Pin<&mut MLIRContext>, value: bool) -> *const c_void;
        #[must_use]
        unsafe fn DenseElementsAttr_get(
            shaped_type: *const c_void, // any type implementing ShapedType trait.
            // Attribute
            values: &[*const c_void],
        ) -> *const c_void;
        #[must_use]
        unsafe fn DictionaryAttr_get(
            context: Pin<&mut MLIRContext>,
            values: &[*const NamedAttribute],
        ) -> *const c_void;
    }
}

#[cfg(test)]
mod tests {
    use crate::IR::{Attributes::Attribute_print, MLIRContext::MLIRContext};

    use super::*;

    #[test]
    fn string_attr_new() {
        let mut context = MLIRContext::new();
        let string_attr = StringAttr_get(context.pin_mut(), "hello_world");
        assert_eq!(
            "\"hello_world\"",
            unsafe { Attribute_print(string_attr) }.as_str()
        );
    }
}
