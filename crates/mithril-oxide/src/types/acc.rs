use super::Type;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct DataBoundsType<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Type<'c> for DataBoundsType<'c> {
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

impl<'c> Clone for DataBoundsType<'c> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<'c> fmt::Display for DataBoundsType<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
