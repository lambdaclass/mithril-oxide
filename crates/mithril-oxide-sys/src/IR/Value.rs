pub use self::ffi::{Value_dump, Value_print};

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Value.hpp");

        type c_void;

        #[must_use]
        pub unsafe fn Value_print(value: *mut c_void) -> String;

        pub unsafe fn Value_dump(value: *mut c_void);
    }
}

#[cfg(test)]
mod tests {}
