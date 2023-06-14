use cxx::UniquePtr;
pub use ffi::{
    AffineExpr, BaseMemRefType, FloatType, FunctionType, IndexType, IntegerType, MemRefType,
    RankedTensorType, TensorType, VectorType,
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

macro_rules! impl_type_conversion {
    ($type_name:ident, $func_name:ident) => {
        impl From<&ffi::$type_name> for UniquePtr<crate::IR::Value::ffi::Type> {
            fn from(val: &ffi::$type_name) -> Self {
                crate::IR::Types::ffi::$func_name(val)
            }
        }
    };
}

impl_type_conversion!(FunctionType, FunctionType_to_Type);
impl_type_conversion!(IntegerType, IntegerType_to_Type);
impl_type_conversion!(FloatType, FloatType_to_Type);
impl_type_conversion!(TensorType, TensorType_to_Type);
impl_type_conversion!(BaseMemRefType, BaseMemRefType_to_Type);
impl_type_conversion!(MemRefType, MemRefType_to_Type);
impl_type_conversion!(RankedTensorType, RankedTensorType_to_Type);
impl_type_conversion!(VectorType, VectorType_to_Type);
impl_type_conversion!(IndexType, IndexType_to_Type);
