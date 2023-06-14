pub use ffi::Type;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Types.hpp");

        type Type;
        type Value = crate::IR::Value::Value;

        type ShapedType = crate::IR::BuiltinTypes::ShapedType;
        type FunctionType = crate::IR::BuiltinTypes::FunctionType;
        type IntegerType = crate::IR::BuiltinTypes::IntegerType;
        type FloatType = crate::IR::BuiltinTypes::FloatType;
        type TensorType = crate::IR::BuiltinTypes::TensorType;
        type BaseMemRefType = crate::IR::BuiltinTypes::BaseMemRefType;
        type MemRefType = crate::IR::BuiltinTypes::MemRefType;
        type RankedTensorType = crate::IR::BuiltinTypes::RankedTensorType;
        type VectorType = crate::IR::BuiltinTypes::VectorType;
        type IndexType = crate::IR::BuiltinTypes::IndexType;

        #[must_use]
        pub fn isIndex(self: &Type) -> bool;
        #[must_use]
        pub fn isFloat8E5M2(self: &Type) -> bool;
        #[must_use]
        pub fn isFloat8E4M3FN(self: &Type) -> bool;
        #[must_use]
        pub fn isBF16(self: &Type) -> bool;
        #[must_use]
        pub fn isF16(self: &Type) -> bool;
        #[must_use]
        pub fn isF32(self: &Type) -> bool;
        #[must_use]
        pub fn isF64(self: &Type) -> bool;
        #[must_use]
        pub fn isF80(self: &Type) -> bool;
        #[must_use]
        pub fn isF128(self: &Type) -> bool;

        /// Return true if this is an integer type with the specified width.
        #[must_use]
        pub fn isInteger(self: &Type, width: u32) -> bool;
        /// Return true if this is a signless integer type (with the specified width).
        #[must_use]
        pub fn isSignlessInteger(self: &Type) -> bool;
        #[rust_name = "isSignlessIntegerOfWidth"]
        #[must_use]
        pub fn isSignlessInteger(self: &Type, width: u32) -> bool;
        /// Return true if this is a signed integer type (with the specified width).
        #[must_use]
        pub fn isSignedInteger(self: &Type) -> bool;
        #[rust_name = "isSignedIntegerOfWidth"]
        #[must_use]
        pub fn isSignedInteger(self: &Type, width: u32) -> bool;
        /// Return true if this is an unsigned integer type (with the specified
        /// width).
        #[must_use]
        pub fn isUnsignedInteger(self: &Type) -> bool;
        #[rust_name = "isUnsignedIntegerOfWidth"]
        #[must_use]
        pub fn isUnsignedInteger(self: &Type, width: u32) -> bool;

        /// Return the bit width of an integer or a float type, assert failure on
        /// other types.
        #[must_use]
        pub fn getIntOrFloatBitWidth(self: &Type) -> u32;

        /// Return true if this is a signless integer or index type.
        #[must_use]
        pub fn isSignlessIntOrIndex(self: &Type) -> bool;
        /// Return true if this is a signless integer, index, or float type.
        #[must_use]
        pub fn isSignlessIntOrIndexOrFloat(self: &Type) -> bool;
        /// Return true of this is a signless integer or a float type.
        #[must_use]
        pub fn isSignlessIntOrFloat(self: &Type) -> bool;

        /// Return true if this is an integer (of any signedness) or an index type.
        #[must_use]
        pub fn isIntOrIndex(self: &Type) -> bool;
        /// Return true if this is an integer (of any signedness) or a float type.
        #[must_use]
        pub fn isIntOrFloat(self: &Type) -> bool;
        /// Return true if this is an integer (of any signedness), index, or float
        /// type.
        #[must_use]
        pub fn isIntOrIndexOrFloat(self: &Type) -> bool;

        pub fn dump(self: &Type);
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Types.hpp");

        // Value related
        pub fn Value_getType(value: &Value) -> UniquePtr<Type>;

        pub fn FunctionType_to_Type(value: &FunctionType) -> UniquePtr<Type>;
        pub fn IntegerType_to_Type(value: &IntegerType) -> UniquePtr<Type>;
        pub fn FloatType_to_Type(value: &FloatType) -> UniquePtr<Type>;
        pub fn TensorType_to_Type(value: &TensorType) -> UniquePtr<Type>;
        pub fn BaseMemRefType_to_Type(value: &BaseMemRefType) -> UniquePtr<Type>;
        pub fn MemRefType_to_Type(value: &MemRefType) -> UniquePtr<Type>;
        pub fn RankedTensorType_to_Type(value: &RankedTensorType) -> UniquePtr<Type>;
        pub fn VectorType_to_Type(value: &VectorType) -> UniquePtr<Type>;
        pub fn IndexType_to_Type(value: &IndexType) -> UniquePtr<Type>;
        pub fn ShapedType_to_Type(value: &ShapedType) -> UniquePtr<Type>;
    }
}

impl fmt::Debug for ffi::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Type").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    // use super::*;
}
