pub use self::ffi::TokenType_get;

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/MLProgram/IR/MLProgram.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys::ml_program"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/MLProgram/IR/MLProgram.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        pub unsafe fn TokenType_get(type_: *const c_void) -> *const c_void;
    }
}
