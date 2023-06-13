pub use self::ffi::Region;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Region.hpp");

        type Region;
    }
}

impl fmt::Debug for ffi::Region {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Region").finish_non_exhaustive()
    }
}
