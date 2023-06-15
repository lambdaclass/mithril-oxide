pub use ffi::{FunctionType_get, IntegerType_get};

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinTypes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinTypes.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        // Constructors

        /// returns a type.
        pub fn IntegerType_get(
            context: Pin<&mut MLIRContext>,
            width: u32,
            has_sign: bool,
            is_signed: bool,
        ) -> *const c_void;

        /// # Safety
        /// - inputs: pointer to a valid Type
        /// - results: pointer to a valid Type
        /// - returns a type.
        pub unsafe fn FunctionType_get(
            context: Pin<&mut MLIRContext>,
            inputs: &[*const c_void],
            results: &[*const c_void],
        ) -> *const c_void;
    }
}
