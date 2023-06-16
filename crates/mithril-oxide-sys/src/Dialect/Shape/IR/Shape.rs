pub use self::ffi::DeviceAsyncTokenType_get;

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Shape/IR/Shape.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys::pdl"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Shape/IR/Shape.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        pub unsafe fn ShapeType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn SizeType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn ValueShapeType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn WitnessType_get(context: *mut MLIRContext) -> *const c_void;
    }
}
