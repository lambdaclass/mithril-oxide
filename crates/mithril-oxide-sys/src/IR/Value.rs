use cxx::UniquePtr;

pub use self::ffi::Value;
use self::ffi::{Type, Value_print};
use std::{fmt, pin::Pin};

use super::Types::ffi::Value_getType;

#[cxx::bridge]
pub(crate) mod ffi {

    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Value.hpp");

        type Value;
        type BlockArgument = crate::IR::Block::BlockArgument;
        type OpResult = crate::IR::Operation::OpResult;
        type Type = crate::IR::Types::Type;

        pub fn dump(self: Pin<&mut Value>);
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Value.hpp");

        fn BlockArgument_toValue(block: &BlockArgument) -> UniquePtr<Value>;
        fn OpResult_to_Value(block: &OpResult) -> UniquePtr<Value>;

        #[must_use]
        fn Value_print(op: Pin<&mut Value>) -> String;
    }
}

impl ffi::Value {
    #[must_use]
    pub fn get_type(&self) -> UniquePtr<Type> {
        Value_getType(self)
    }

    #[must_use]
    pub fn print(self: Pin<&mut Self>) -> String {
        Value_print(self)
    }
}

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

impl From<&ffi::OpResult> for UniquePtr<ffi::Value> {
    fn from(val: &ffi::OpResult) -> Self {
        ffi::OpResult_to_Value(val)
    }
}

#[cfg(test)]
mod tests {}
