pub use self::ffi::Block;
use self::ffi::{Block_addArgument, Location, Type};
use std::fmt;
use std::pin::Pin;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Block.hpp");

        type Block;
        // type BlockArgument;

        type Region = crate::IR::Region::Region;
        type Operation = crate::IR::Operation::Operation;
        type Type = crate::IR::Types::Type;
        type Location = crate::IR::Location::Location;

        #[must_use]
        pub fn getParent(self: &Block) -> *mut Region;
        #[must_use]
        pub fn getParentOp(self: Pin<&mut Block>) -> *mut Operation;
        #[must_use]
        pub fn isEntryBlock(self: Pin<&mut Block>) -> bool;
        pub fn erase(self: Pin<&mut Block>);
        #[must_use]
        pub fn getNumArguments(self: Pin<&mut Block>) -> u32;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Block.hpp");

        fn Block_addArgument(block: Pin<&mut Block>, ttype: &Type, loc: &Location);
    }
}

impl ffi::Block {
    pub fn add_argument(self: Pin<&mut Self>, r#type: &Type, loc: &Location) {
        Block_addArgument(self, r#type, loc);
    }
}

impl fmt::Debug for ffi::Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Block").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {}
