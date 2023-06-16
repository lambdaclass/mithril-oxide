pub use self::ffi::{ShapeType_get, SizeType_get, ValueShapeType_get, WitnessType_get};

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Shape/IR/Shape.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        type c_void = crate::c_void;
    }

    #[namespace = "mithril_oxide_sys::shape"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/Shape/IR/Shape.hpp");

        pub unsafe fn ShapeType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn SizeType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn ValueShapeType_get(context: *mut MLIRContext) -> *const c_void;
        pub unsafe fn WitnessType_get(context: *mut MLIRContext) -> *const c_void;
    }
}
