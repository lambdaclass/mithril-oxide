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

        type c_void = crate::c_void;

        // Constructors

        /// returns a type.
        #[must_use]
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
        #[must_use]
        pub unsafe fn FunctionType_get(
            context: Pin<&mut MLIRContext>,
            inputs: &[*const c_void],
            results: &[*const c_void],
        ) -> *const c_void;
    }
}

#[cfg(test)]
mod tests {
    use crate::IR::{MLIRContext::MLIRContext, Types::Type_print};

    use super::*;

    #[test]
    fn function_type() {
        let mut context = MLIRContext::new();

        unsafe {
            let a = IntegerType_get(context.pin_mut(), 1, false, false);
            let a_str = Type_print(a);
            assert_eq!("i1", a_str.as_str());

            let b = IntegerType_get(context.pin_mut(), 8, false, false);
            let b_str = Type_print(b);
            assert_eq!("i8", b_str.as_str());

            let func_type = FunctionType_get(context.pin_mut(), &[a], &[b]);
            let func_str = Type_print(func_type);
            assert_eq!("(i1) -> i8", func_str.as_str());
        }
    }
}
