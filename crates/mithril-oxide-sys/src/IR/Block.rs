use self::ffi::{c_void, Block_addArgument, Block_getArgument, Location};
pub use self::ffi::{Block, BlockArgument};
use std::fmt;
use std::pin::Pin;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Block.hpp");

        type Block;
        type BlockArgument;

        type Region = crate::IR::Region::Region;
        type Operation = crate::IR::Operation::Operation;
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

        pub fn dump(self: Pin<&mut Block>);
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Block.hpp");

        type c_void = crate::c_void;

        unsafe fn Block_addArgument(block: Pin<&mut Block>, ttype: *const c_void, loc: &Location);
        #[must_use]
        fn Block_getArgument(block: Pin<&mut Block>, i: u32) -> *mut c_void;
    }
}

impl ffi::Block {
    // type is a Type.
    pub unsafe fn add_argument(self: Pin<&mut Self>, r#type: *const c_void, loc: &Location) {
        Block_addArgument(self, r#type, loc);
    }

    #[must_use]
    /// returns a pointer to a `BlockArgument` / `Value`
    pub fn get_argument(self: Pin<&mut Self>, i: u32) -> *mut c_void {
        Block_getArgument(self, i)
    }
}

impl fmt::Debug for ffi::Block {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Block").finish_non_exhaustive()
    }
}

impl fmt::Debug for ffi::BlockArgument {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BlockArgument").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {}
