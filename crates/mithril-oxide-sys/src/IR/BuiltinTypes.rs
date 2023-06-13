pub use self::ffi::FunctionType;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/BuiltinTypes.hpp");

        type FunctionType;
        type IntegerType;
        type FloatType;
        type TensorType;
        type BaseMemRefType;
        type MemRefType;
        type RankedTensorType;
        type VectorType;
        type AffineExpr;
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
