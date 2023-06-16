pub use self::ffi::{OpaqueType_get, PointerType_get};

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/EmitC/IR/EmitC.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        type c_void = crate::c_void;
    }

    #[namespace = "mithril_oxide_sys::emitc"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/EmitC/IR/EmitC.hpp");

        pub unsafe fn OpaqueType_get(context: *mut MLIRContext, name: &str) -> *const c_void;
        #[must_use]
        pub unsafe fn PointerType_get(type_: *const c_void) -> *const c_void;
    }
}
