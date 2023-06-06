use super::Attribute;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct OpaqueAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for OpaqueAttr<'c> {}

impl<'c> fmt::Display for OpaqueAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
