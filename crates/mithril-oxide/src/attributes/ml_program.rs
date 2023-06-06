use super::Attribute;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct ExternAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ExternAttr<'c> {}

impl<'c> fmt::Display for ExternAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
