use super::Type;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct BFloat16Type<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for BFloat16Type<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for BFloat16Type<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for BFloat16Type<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ComplexType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for ComplexType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for ComplexType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for ComplexType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct Float8E4M3B11FNUZType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for Float8E4M3B11FNUZType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for Float8E4M3B11FNUZType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for Float8E4M3B11FNUZType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct Float8E4M3FNType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for Float8E4M3FNType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for Float8E4M3FNType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for Float8E4M3FNType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct Float8E4M3FNUZType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for Float8E4M3FNUZType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for Float8E4M3FNUZType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for Float8E4M3FNUZType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct Float8E5M2Type<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for Float8E5M2Type<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for Float8E5M2Type<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for Float8E5M2Type<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct Float8E5M2FNUZType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for Float8E5M2FNUZType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for Float8E5M2FNUZType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for Float8E5M2FNUZType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct Float16Type<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for Float16Type<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for Float16Type<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for Float16Type<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct Float32Type<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for Float32Type<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for Float32Type<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for Float32Type<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct Float64Type<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for Float64Type<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for Float64Type<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for Float64Type<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct Float80Type<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for Float80Type<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for Float80Type<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for Float80Type<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct Float128Type<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for Float128Type<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for Float128Type<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for Float128Type<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct FunctionType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for FunctionType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for FunctionType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for FunctionType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct IndexType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for IndexType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for IndexType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for IndexType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct IntegerType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for IntegerType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for IntegerType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for IntegerType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct MemRefType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for MemRefType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for MemRefType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for MemRefType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct NoneType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for NoneType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for NoneType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for NoneType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct OpaqueType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for OpaqueType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for OpaqueType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for OpaqueType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct RankedTensorType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for RankedTensorType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for RankedTensorType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for RankedTensorType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct TupleType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for TupleType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for TupleType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for TupleType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct UnrankedMemRefType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for UnrankedMemRefType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for UnrankedMemRefType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for UnrankedMemRefType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct UnrankedTensorType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for UnrankedTensorType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for UnrankedTensorType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for UnrankedTensorType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct VectorType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for VectorType<'c> {
    fn size(&self) -> usize {
        todo!()
    }

    fn size_in_bits(&self) -> usize {
        todo!()
    }

    fn abi_alignment(&self) -> usize {
        todo!()
    }

    fn preferred_alignment(&self) -> usize {
        todo!()
    }
}

impl<'c> Clone for VectorType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for VectorType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
