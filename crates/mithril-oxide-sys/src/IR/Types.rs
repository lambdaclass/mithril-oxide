pub use ffi::{Type_dump, Type_getIntOrFloatBitWidth, Type_print, Value_getType};

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Types.hpp");

        type c_void = crate::c_void;

        /// value must be a mlir Value
        ///
        /// returns a Type
        #[must_use]
        pub unsafe fn Value_getType(value: *const c_void) -> *const c_void;

        /// Return the bit width of an integer or a float type, assert failure on
        /// other types.
        #[must_use]
        pub unsafe fn Type_getIntOrFloatBitWidth(type_ptr: *const c_void) -> u32;

        pub unsafe fn Type_dump(type_ptr: *const c_void);

        #[must_use]
        pub unsafe fn Type_print(type_ptr: *const c_void) -> String;

        // todo: more utility methods?
    }
}
