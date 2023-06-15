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

        unsafe fn StringAttr_get(context: Pin<&mut MLIRContext>, value: &str) -> *const c_void;
        unsafe fn IntegerAttr_get(context: Pin<&mut MLIRContext>, value: &str) -> *const c_void;
        unsafe fn BoolAttr_get(context: Pin<&mut MLIRContext>, value: bool) -> *const c_void;
        unsafe fn DenseElementsAttr_get(
            shaped_type: *const c_void, // any type implementing ShapedType trait.
            // Attribute
            values: &[*const c_void],
        ) -> *const c_void;
        unsafe fn DictionaryAttr_get(
            context: Pin<&mut MLIRContext>,
            values: &[*const NamedAttribute],
        ) -> *const c_void;
    }
}

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
