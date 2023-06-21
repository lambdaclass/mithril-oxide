pub use self::ffi::{CallSiteLoc_get, FileLineColLoc_get, UnknownLoc_get};

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Location.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Location.hpp");

        type c_void = crate::c_void;

        #[must_use]
        pub unsafe fn UnknownLoc_get(ctx: Pin<&mut MLIRContext>) -> *const c_void;

        // filename - stringattr
        #[must_use]
        pub unsafe fn FileLineColLoc_get(
            filename: *const c_void,
            line: u32,
            column: u32,
        ) -> *const c_void;

        // callee - location
        // caller - location
        #[must_use]
        #[allow(clippy::similar_names)]
        pub unsafe fn CallSiteLoc_get(
            callee_ptr: *const c_void,
            caller_ptr: *const c_void,
        ) -> *const c_void;
    }
}
