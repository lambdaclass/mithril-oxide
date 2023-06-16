pub use self::ffi::{
    ArrayAttr_get, BoolAttr_get, DenseBoolArrayAttr_get, DenseElementsAttr_get,
    DenseF32ArrayAttr_get, DenseF64ArrayAttr_get, DenseFPElementsAttr_get, DenseI16ArrayAttr_get,
    DenseI32ArrayAttr_get, DenseI64ArrayAttr_get, DenseI8ArrayAttr_get, DenseIntElementsAttr_get,
    DictionaryAttr_get, FlatSymbolRefAttr_get, IntegerAttr_get, OpaqueAttr_get,
    SparseElementsAttr_get, StridedLayoutAttr_get, StringAttr_get, TypeAttr_get, UnitAttr_get,
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

        type c_void = crate::c_void;

        #[must_use]
        fn UnitAttr_get(context: Pin<&mut MLIRContext>) -> *const c_void;
        #[must_use]
        unsafe fn TypeAttr_get(ttype: *const c_void) -> *const c_void;
        #[must_use]
        fn StringAttr_get(context: Pin<&mut MLIRContext>, value: &str) -> *const c_void;
        #[must_use]
        fn FlatSymbolRefAttr_get(context: Pin<&mut MLIRContext>, value: &str) -> *const c_void;
        #[must_use]
        fn IntegerAttr_get(context: Pin<&mut MLIRContext>, value: &str) -> *const c_void;
        #[must_use]
        fn BoolAttr_get(context: Pin<&mut MLIRContext>, value: bool) -> *const c_void;

        #[must_use]
        unsafe fn StridedLayoutAttr_get(
            context: Pin<&mut MLIRContext>,
            offset: i64,
            strides: &[i64],
        ) -> *const c_void;

        #[must_use]
        unsafe fn OpaqueAttr_get(
            dialect: *const c_void, // StringAttr
            attr_data: &str,
            ttype: *const c_void, // Type
        ) -> *const c_void;

        #[must_use]
        unsafe fn SparseElementsAttr_get(
            shaped_type: *const c_void, // ShapedType
            indices: *const c_void,     // DenseElementsAttr
            values: *const c_void,      // DenseElementsAttr
        ) -> *const c_void;

        #[must_use]
        unsafe fn DenseElementsAttr_get(
            shaped_type: *const c_void, // any type implementing ShapedType trait.
            // Attribute
            values: &[*const c_void],
        ) -> *const c_void;
        #[must_use]
        unsafe fn DenseFPElementsAttr_get(
            shaped_type: *const c_void, // any type implementing ShapedType trait.
            // Attribute
            values: &[*const c_void],
        ) -> *const c_void;
        #[must_use]
        unsafe fn DenseIntElementsAttr_get(
            shaped_type: *const c_void, // any type implementing ShapedType trait.
            // Attribute
            values: &[*const c_void],
        ) -> *const c_void;
        #[must_use]
        unsafe fn DictionaryAttr_get(
            context: Pin<&mut MLIRContext>,
            values: &[*const NamedAttribute],
        ) -> *const c_void;

        #[must_use]
        unsafe fn ArrayAttr_get(
            context: Pin<&mut MLIRContext>,
            // Attribute
            values: &[*const c_void],
        ) -> *const c_void;

        #[must_use]
        unsafe fn DenseBoolArrayAttr_get(
            context: Pin<&mut MLIRContext>,
            values: &[bool],
        ) -> *const c_void;

        #[must_use]
        unsafe fn DenseI8ArrayAttr_get(
            context: Pin<&mut MLIRContext>,
            values: &[i8],
        ) -> *const c_void;

        #[must_use]
        unsafe fn DenseI16ArrayAttr_get(
            context: Pin<&mut MLIRContext>,
            values: &[i16],
        ) -> *const c_void;

        #[must_use]
        unsafe fn DenseI32ArrayAttr_get(
            context: Pin<&mut MLIRContext>,
            values: &[i32],
        ) -> *const c_void;

        #[must_use]
        unsafe fn DenseI64ArrayAttr_get(
            context: Pin<&mut MLIRContext>,
            values: &[i64],
        ) -> *const c_void;

        #[must_use]
        unsafe fn DenseF32ArrayAttr_get(
            context: Pin<&mut MLIRContext>,
            values: &[f32],
        ) -> *const c_void;

        #[must_use]
        unsafe fn DenseF64ArrayAttr_get(
            context: Pin<&mut MLIRContext>,
            values: &[f64],
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
