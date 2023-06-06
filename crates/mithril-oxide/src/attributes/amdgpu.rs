use super::Attribute;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct MFMAPermBAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for MFMAPermBAttr<'c> {}

impl<'c> fmt::Display for MFMAPermBAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
