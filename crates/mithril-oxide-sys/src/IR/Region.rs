pub use self::ffi::Region;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Region.hpp");

        type Block = crate::IR::Block::Block;
        type Region;

        #[must_use]
        fn emplaceBlock(self: Pin<&mut Region>) -> Pin<&mut Block>;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Region.hpp");
    }
}

impl ffi::Region {}

impl fmt::Debug for ffi::Region {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Region").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {}
