use cxx::UniquePtr;

pub use self::ffi::Value;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Value.hpp");

        type Value;
        type BlockArgument = crate::IR::Block::BlockArgument;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Value.hpp");

        fn BlockArgument_toValue(block: &BlockArgument) -> UniquePtr<Value>;
    }
}

impl ffi::Value {}

impl fmt::Debug for ffi::Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Value").finish_non_exhaustive()
    }
}

impl From<&ffi::BlockArgument> for UniquePtr<ffi::Value> {
    fn from(val: &ffi::BlockArgument) -> Self {
        ffi::BlockArgument_toValue(val)
    }
}

#[cfg(test)]
mod tests {}
