use self::ffi::{c_void, Operation_getName, Operation_getResult, Operation_print};
pub use self::ffi::{OpResult, Operation};
use std::{fmt, pin::Pin};

#[cxx::bridge]
pub(crate) mod ffi {

    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Operation.hpp");

        type Operation;
        type OpResult;
        type Block = crate::IR::Block::Block;
        type Region = crate::IR::Region::Region;

        #[must_use]
        pub fn getBlock(self: Pin<&mut Operation>) -> *mut Block;
        #[must_use]
        pub fn getParentRegion(self: Pin<&mut Operation>) -> *mut Region;
        #[must_use]
        pub fn getParentOp(self: Pin<&mut Operation>) -> *mut Operation;
        #[must_use]
        pub fn getNumResults(self: Pin<&mut Operation>) -> u32;

        pub fn dump(self: Pin<&mut Operation>);
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Operation.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        #[must_use]
        fn Operation_print(op: Pin<&mut Operation>) -> String;

        #[must_use]
        fn Operation_getName(op: Pin<&mut Operation>) -> &str;

        #[must_use]
        fn Operation_getResult(op: Pin<&mut Operation>, idx: u32) -> *mut c_void;
    }
}

impl ffi::Operation {
    #[must_use]
    pub fn get_name(self: Pin<&mut Self>) -> &str {
        Operation_getName(self)
    }

    #[must_use]
    pub fn print(self: Pin<&mut Self>) -> String {
        Operation_print(self)
    }

    #[must_use]
    pub fn getResult(self: Pin<&mut Self>, idx: u32) -> *mut c_void {
        Operation_getResult(self, idx)
    }
}

impl fmt::Debug for ffi::Operation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Operation").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {}
