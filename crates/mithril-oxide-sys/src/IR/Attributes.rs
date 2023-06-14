use self::ffi::*;
pub use self::ffi::{Attribute, NamedAttribute};
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Attributes.hpp");

        type NamedAttribute;
        type Attribute;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Attributes.hpp");

        #[must_use]
        fn Attribute_print(attribute: &Attribute) -> String;
    }
}

impl fmt::Debug for ffi::NamedAttribute {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("NamedAttribute").finish_non_exhaustive()
    }
}

impl fmt::Debug for ffi::Attribute {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Attribute").finish_non_exhaustive()
    }
}

impl ffi::Attribute {
    #[must_use]
    pub fn print(&self) -> String {
        Attribute_print(self)
    }
}
