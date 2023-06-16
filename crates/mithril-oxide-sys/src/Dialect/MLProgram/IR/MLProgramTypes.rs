pub use self::ffi::TokenType_get;

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/MLProgram/IR/MLProgramTypes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        type c_void = crate::c_void;
    }

    #[namespace = "mithril_oxide_sys::ml_program"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/MLProgram/IR/MLProgramTypes.hpp");

        pub unsafe fn TokenType_get(type_: *mut MLIRContext) -> *const c_void;
    }
}
