use super::Type;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct CoroHandleType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for CoroHandleType<'c> {
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

impl<'c> Clone for CoroHandleType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for CoroHandleType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct CoroIdType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for CoroIdType<'c> {
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

impl<'c> Clone for CoroIdType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for CoroIdType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct CoroStateType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for CoroStateType<'c> {
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

impl<'c> Clone for CoroStateType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for CoroStateType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct GroupType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for GroupType<'c> {
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

impl<'c> Clone for GroupType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for GroupType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct TokenType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for TokenType<'c> {
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

impl<'c> Clone for TokenType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for TokenType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ValueType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for ValueType<'c> {
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

impl<'c> Clone for ValueType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for ValueType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
