use super::Attribute;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct ClauseDefaultValueAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ClauseDefaultValueAttr<'c> {}

impl<'c> fmt::Display for ClauseDefaultValueAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ReductionOperatorAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ReductionOperatorAttr<'c> {}

impl<'c> fmt::Display for ReductionOperatorAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
