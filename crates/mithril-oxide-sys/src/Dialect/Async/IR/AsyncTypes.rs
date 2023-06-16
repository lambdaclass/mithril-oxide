#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Async/IR/AsyncTypes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys::async"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Async/IR/AsyncTypes.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        unsafe fn CoroHandleType_get(context: *mut MLIRContext) -> *const c_void;
        unsafe fn CoroIdType_get(context: *mut MLIRContext) -> *const c_void;
        unsafe fn CoroStateType_get(context: *mut MLIRContext) -> *const c_void;
        unsafe fn GroupType_get(context: *mut MLIRContext) -> *const c_void;
        unsafe fn TokenType_get(context: *mut MLIRContext) -> *const c_void;
        unsafe fn ValueType_get(type_: *const c_void) -> *const c_void;
    }
}
