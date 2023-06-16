pub use self::ffi::{OpaqueType_get, ValueType_get};

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/EmitC/IR/EmitC.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys::async"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Async/IR/AsyncTypes.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        pub unsafe fn OpaqueType_get(context: *mut MLIRContext, name: &str) -> *const c_void;
        pub unsafe fn ValueType_get(type_: *const c_void) -> *const c_void;
    }
}
