#![allow(non_snake_case)]
#![deny(clippy::pedantic)]
#![deny(warnings)]
//
#![allow(clippy::module_name_repetitions)]

pub use cxx::UniquePtr;

pub mod Dialect;
pub mod IR;
pub mod InitAllDialects;
pub mod InitAllTranslations;

macro_rules! impl_conversion {
    ($type_name:ident, $func_name:ident, $target_type:ident) => {
        impl From<&$type_name> for UniquePtr<$target_type> {
            fn from(val: &$type_name) -> Self {
                $func_name(val)
            }
        }
    };
}

pub(crate) use impl_conversion;
