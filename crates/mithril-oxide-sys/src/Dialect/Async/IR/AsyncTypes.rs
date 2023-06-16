pub use self::ffi::{
    CoroHandleType_get, CoroIdType_get, CoroStateType_get, GroupType_get, TokenType_get,
    ValueType_get,
};

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Async/IR/AsyncTypes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        type c_void = crate::c_void;
    }

    #[namespace = "mithril_oxide_sys::async"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Async/IR/AsyncTypes.hpp");

        pub unsafe fn CoroHandleType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn CoroIdType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn CoroStateType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn GroupType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn TokenType_get(context: *mut MLIRContext) -> *const c_void;
        #[must_use]
        pub unsafe fn ValueType_get(type_: *const c_void) -> *const c_void;
    }
}
