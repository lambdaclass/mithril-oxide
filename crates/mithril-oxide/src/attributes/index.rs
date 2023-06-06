use super::Attribute;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct IndexCmpPredicateAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for IndexCmpPredicateAttr<'c> {}

impl<'c> fmt::Display for IndexCmpPredicateAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
