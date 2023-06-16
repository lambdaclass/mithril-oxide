pub use self::ffi::{AnyOpType_get, OperationType_get, ParamType_get};

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Transform/IR/TransformTypes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        type c_void = crate::c_void;
    }

    #[namespace = "mithril_oxide_sys::transform"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Transform/IR/TransformTypes.hpp");

        pub unsafe fn AnyOpType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn OperationType_get(context: *mut MLIRContext, name: &str) -> *const c_void;
        pub unsafe fn ParamType_get(
            context: *mut MLIRContext,
            type_: *const c_void,
        ) -> *const c_void;
    }
}
