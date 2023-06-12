use super::Type;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct ShapeType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for ShapeType<'c> {
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

impl<'c> Clone for ShapeType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for ShapeType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct SizeType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for SizeType<'c> {
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

impl<'c> Clone for SizeType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for SizeType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ValueShapeType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for ValueShapeType<'c> {
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

impl<'c> Clone for ValueShapeType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for ValueShapeType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct WitnessType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for WitnessType<'c> {
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

impl<'c> Clone for WitnessType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for WitnessType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
