pub use self::ffi::DeviceAsyncTokenType_get;

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Transform/IR/TransformTypes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys::sparse_tensor"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Transform/IR/TransformTypes.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        pub unsafe fn AnyOpType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn OperationType_get(context: *mut MLIRContext, name: &str) -> *const c_void;
        pub unsafe fn ParamType_get(
            context: *mut MLIRContext,
            type_: *const c_void,
        ) -> *const c_void;
    }
}
