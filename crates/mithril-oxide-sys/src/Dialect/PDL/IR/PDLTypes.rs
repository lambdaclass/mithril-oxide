pub use self::ffi::{
    AttributeType_get, OperationType_get, RangeType_get, TypeType_get, ValueType_get,
};

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/PDL/IR/PDLTypes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        type c_void = crate::c_void;
    }

    #[namespace = "mithril_oxide_sys::pdl"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/PDL/IR/PDLTypes.hpp");

        pub unsafe fn AttributeType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn OperationType_get(context: *mut MLIRContext) -> *const c_void;
        #[must_use]
        pub unsafe fn RangeType_get(elementType: *const c_void) -> *const c_void;
        pub unsafe fn TypeType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn ValueType_get(context: *mut MLIRContext) -> *const c_void;
    }
}
