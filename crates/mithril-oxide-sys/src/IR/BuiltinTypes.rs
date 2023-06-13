use cxx::UniquePtr;
pub use ffi::{
    AffineExpr, BaseMemRefType, FloatType, IndexType, IntegerType, MemRefType, RankedTensorType,
    TensorType, VectorType, FunctionType
};

use self::ffi::MLIRContext;
use std::{fmt, pin::Pin};

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinTypes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;

        type FunctionType;
        type IntegerType;
        type FloatType;
        type TensorType;
        type BaseMemRefType;
        type MemRefType;
        type RankedTensorType;
        type VectorType;
        type AffineExpr;
        type IndexType;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinTypes.hpp");

        fn IntegerType_get(
            context: Pin<&mut MLIRContext>,
            width: u32,
            has_sign: bool,
            is_signed: bool,
        ) -> UniquePtr<IntegerType>;
    }
}

macro_rules! impl_type_debug {
    ($ident:ident) => {
        impl fmt::Debug for ffi::$ident {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.debug_struct(stringify!($ident)).finish_non_exhaustive()
            }
        }
    };
}

impl_type_debug!(FunctionType);
impl_type_debug!(IntegerType);
impl_type_debug!(FloatType);
impl_type_debug!(TensorType);
impl_type_debug!(BaseMemRefType);
impl_type_debug!(MemRefType);
impl_type_debug!(RankedTensorType);
impl_type_debug!(VectorType);
impl_type_debug!(AffineExpr);
impl_type_debug!(IndexType);

impl ffi::IntegerType {
    #[must_use]
    pub fn new(
        ctx: Pin<&mut MLIRContext>,
        width: u32,
        has_sign: bool,
        is_signed: bool,
    ) -> UniquePtr<Self> {
        ffi::IntegerType_get(ctx, width, has_sign, is_signed)
    }
}
impl From<&ffi::IntegerType> for UniquePtr<crate::IR::Value::ffi::Type> {
    fn from(val: &ffi::IntegerType) -> Self {
        crate::IR::Types::ffi::IntegerType_to_Type(val)
    }
}


macro_rules! impl_type_conversion {
    ($ident:ident) => {
        impl From<&ffi::$ident> for UniquePtr<crate::IR::Value::ffi::Type> {
            fn from(val: &ffi::$ident) -> Self {
                concat_idents!(crate::IR::Types::ffi::$ident, _to_Type(val))
            }
        }
    };
}
