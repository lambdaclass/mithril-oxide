pub use self::ffi::NamedAttribute;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Attributes.hpp");

        type NamedAttribute;
    }
}

impl fmt::Debug for ffi::NamedAttribute {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("NamedAttribute").finish_non_exhaustive()
    }
}
