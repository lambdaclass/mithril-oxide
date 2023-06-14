use cxx::UniquePtr;

use self::ffi::{Attribute_print, NamedAttribute_getName, NamedAttribute_getValue, NamedAttribute_new, StringAttr};
pub use self::ffi::{Attribute, NamedAttribute};
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Attributes.hpp");

        type NamedAttribute;
        type Attribute;
        type StringAttr = crate::IR::BuiltinAttributes::StringAttr;
        // type MLIRContext = crate::IR::MLIRContext::MLIRContext;

        pub fn dump(self: &Attribute);
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Attributes.hpp");

        #[must_use]
        fn Attribute_print(attribute: &Attribute) -> String;

        #[must_use]
        fn NamedAttribute_new(name: &StringAttr, attr: &Attribute) -> UniquePtr<NamedAttribute>;
        #[must_use]
        fn NamedAttribute_getName(attribute: &NamedAttribute) -> &str;
        #[must_use]
        fn NamedAttribute_getValue(attribute: &NamedAttribute) -> UniquePtr<Attribute>;
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

impl ffi::NamedAttribute {
    #[must_use]
    pub fn new(name: &StringAttr, attr: &Attribute) -> UniquePtr<NamedAttribute> {
        NamedAttribute_new(name, attr)
    }

    #[must_use]
    pub fn get_name(&self) -> &str {
        NamedAttribute_getName(self)
    }

    #[must_use]
    pub fn get_value(&self) -> UniquePtr<Attribute> {
        NamedAttribute_getValue(self)
    }
}
