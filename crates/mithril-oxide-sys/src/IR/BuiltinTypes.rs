use cxx::UniquePtr;
pub use ffi::{
    AffineExpr, BaseMemRefType, FloatType, FunctionType, IndexType, IntegerType, MemRefType,
    RankedTensorType, ShapedType, TensorType, VectorType,
};

use self::ffi::{MLIRContext, Type};
use std::{fmt, pin::Pin};

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinTypes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
        type Type = crate::IR::Types::Type;

        type ShapedType;
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

        // Conversions
        pub fn VectorType_to_ShapedType(value: &VectorType) -> UniquePtr<ShapedType>;
        pub fn MemRefType_to_ShapedType(value: &MemRefType) -> UniquePtr<ShapedType>;
        pub fn TensorType_to_ShapedType(value: &TensorType) -> UniquePtr<ShapedType>;
        pub fn RankedTensorType_to_ShapedType(value: &RankedTensorType) -> UniquePtr<ShapedType>;

        // Constructors
        fn IntegerType_get(
            context: Pin<&mut MLIRContext>,
            width: u32,
            has_sign: bool,
            is_signed: bool,
        ) -> UniquePtr<IntegerType>;

        fn FunctionType_get(
            context: Pin<&mut MLIRContext>,
            inputs: &[*const Type],
            results: &[*const Type],
        ) -> UniquePtr<FunctionType>;
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

impl_type_debug!(ShapedType);
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

impl ffi::FunctionType {
    #[must_use]
    pub fn new(
        ctx: Pin<&mut MLIRContext>,
        inputs: impl IntoIterator<Item = &'a Type>,
        results: impl IntoIterator<Item = &'a Type>,
    ) -> UniquePtr<Self> {
        let inputs_vec = inputs
            .into_iter()
            .map(|x| x as *const _)
            .collect::<Vec<_>>();
        let results_vec = results
            .into_iter()
            .map(|x| x as *const _)
            .collect::<Vec<_>>();
        ffi::FunctionType_get(ctx, &inputs_vec, &results_vec)
    }
}

macro_rules! impl_type_conversion {
    ($type_name:ident, $func_name:ident) => {
        impl From<&ffi::$type_name> for UniquePtr<crate::IR::Types::ffi::Type> {
            fn from(val: &ffi::$type_name) -> Self {
                crate::IR::Types::ffi::$func_name(val)
            }
        }
    };
}

macro_rules! impl_shaped_type_conversion {
    ($type_name:ident, $func_name:ident) => {
        impl From<&ffi::$type_name> for UniquePtr<crate::IR::BuiltinTypes::ffi::ShapedType> {
            fn from(val: &ffi::$type_name) -> Self {
                crate::IR::BuiltinTypes::ffi::$func_name(val)
            }
        }
    };
}

impl_type_conversion!(ShapedType, ShapedType_to_Type);
impl_type_conversion!(FunctionType, FunctionType_to_Type);
impl_type_conversion!(IntegerType, IntegerType_to_Type);
impl_type_conversion!(FloatType, FloatType_to_Type);
impl_type_conversion!(TensorType, TensorType_to_Type);
impl_type_conversion!(BaseMemRefType, BaseMemRefType_to_Type);
impl_type_conversion!(MemRefType, MemRefType_to_Type);
impl_type_conversion!(RankedTensorType, RankedTensorType_to_Type);
impl_type_conversion!(VectorType, VectorType_to_Type);
impl_type_conversion!(IndexType, IndexType_to_Type);

impl_shaped_type_conversion!(TensorType, TensorType_to_ShapedType);
impl_shaped_type_conversion!(MemRefType, MemRefType_to_ShapedType);
impl_shaped_type_conversion!(RankedTensorType, RankedTensorType_to_ShapedType);
impl_shaped_type_conversion!(VectorType, VectorType_to_ShapedType);
