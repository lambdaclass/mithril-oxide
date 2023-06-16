pub use self::ffi::DeviceAsyncTokenType_get;

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/PDL/IR/PDLTypes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys::pdl"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/PDL/IR/PDLTypes.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        pub unsafe fn AttributeType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn OperationType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn RangeType_get(elementType: *const void) -> *const c_void;
        pub unsafe fn TypeType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn ValueType_get(context: *mut MLIRContext) -> *const c_void;
    }
}
