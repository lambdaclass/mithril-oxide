pub use self::ffi::Operation;
use self::ffi::Operation_getName;
use std::{fmt, pin::Pin};

#[cxx::bridge]
pub(crate) mod ffi {

    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Operation.hpp");

        type Operation;
        type Block = crate::IR::Block::Block;
        type Region = crate::IR::Region::Region;

        #[must_use]
        pub fn getBlock(self: Pin<&mut Operation>) -> *mut Block;
        #[must_use]
        pub fn getParentRegion(self: Pin<&mut Operation>) -> *mut Region;
        #[must_use]
        pub fn getParentOp(self: Pin<&mut Operation>) -> *mut Operation;

        pub fn dump(self: Pin<&mut Operation>);
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Operation.hpp");

        #[must_use]
        pub fn Operation_getName(op: Pin<&mut Operation>) -> &str;
    }
}

impl ffi::Operation {
    #[must_use]
    pub fn get_name(self: Pin<&mut Self>) -> &str {
        Operation_getName(self)
    }
}

impl fmt::Debug for ffi::Operation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Operation").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {}
