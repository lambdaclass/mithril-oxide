pub use self::ffi::FunctionType;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinTypes.hpp");

        type FunctionType;
    }
}

impl fmt::Debug for ffi::FunctionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FunctionType").finish_non_exhaustive()
    }
}
