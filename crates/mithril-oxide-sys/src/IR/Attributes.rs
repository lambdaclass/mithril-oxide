use cxx::UniquePtr;

pub use self::ffi::{Attribute_print, NamedAttribute};
use self::ffi::{NamedAttribute_getName, NamedAttribute_getValue, NamedAttribute_new};
use std::fmt;

use super::Value::ffi::c_void;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Attributes.hpp");

        type NamedAttribute;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Attributes.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        #[must_use]
        unsafe fn Attribute_print(attribute: *const c_void) -> String;

        #[must_use]
        /// name: StringAttribute, attr: Attribute
        unsafe fn NamedAttribute_new(
            name: *const c_void,
            attr: *const c_void,
        ) -> UniquePtr<NamedAttribute>;
        #[must_use]
        fn NamedAttribute_getName(attribute: &NamedAttribute) -> &str;
        #[must_use]
        fn NamedAttribute_getValue(attribute: &NamedAttribute) -> *const c_void;
    }
}

impl fmt::Debug for ffi::NamedAttribute {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("NamedAttribute").finish_non_exhaustive()
    }
}

impl ffi::NamedAttribute {
    #[must_use]
    /// name: StringAttribute, attr: Attribute
    pub unsafe fn new(name: *const c_void, attr: *const c_void) -> UniquePtr<NamedAttribute> {
        NamedAttribute_new(name, attr)
    }

    #[must_use]
    pub fn get_name(&self) -> &str {
        NamedAttribute_getName(self)
    }

    #[must_use]
    pub fn get_value(&self) -> *const c_void {
        NamedAttribute_getValue(self)
    }
}
