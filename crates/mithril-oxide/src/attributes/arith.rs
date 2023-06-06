use super::Attribute;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct FastMathFlagsAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for FastMathFlagsAttr<'c> {}

impl<'c> fmt::Display for FastMathFlagsAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
