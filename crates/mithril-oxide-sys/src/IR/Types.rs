pub use ffi::Type;
use std::fmt;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Types.hpp");

        type Type;

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
    }
}

impl fmt::Debug for ffi::Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Type").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
