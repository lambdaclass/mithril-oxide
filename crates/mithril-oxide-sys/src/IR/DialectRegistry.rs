use cxx::UniquePtr;
pub use ffi::DialectRegistry;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/DialectRegistry.hpp");

        type DialectRegistry;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/DialectRegistry.hpp");

        fn DialectRegistry_new() -> UniquePtr<DialectRegistry>;
    }
}

impl ffi::DialectRegistry {
    #[must_use]
    pub fn new() -> UniquePtr<Self> {
        ffi::DialectRegistry_new()
    }
}

impl fmt::Debug for ffi::DialectRegistry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("DialectRegistry").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let _reg = DialectRegistry::new();
    }
}
