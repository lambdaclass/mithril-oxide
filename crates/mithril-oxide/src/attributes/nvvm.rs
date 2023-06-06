use super::Attribute;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct MMAB1OpAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for MMAB1OpAttr<'c> {}

impl<'c> fmt::Display for MMAB1OpAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct MMAFragAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for MMAFragAttr<'c> {}

impl<'c> fmt::Display for MMAFragAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct MMAIntOverflowAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for MMAIntOverflowAttr<'c> {}

impl<'c> fmt::Display for MMAIntOverflowAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct MMALayoutAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for MMALayoutAttr<'c> {}

impl<'c> fmt::Display for MMALayoutAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct MMAShapeAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for MMAShapeAttr<'c> {}

impl<'c> fmt::Display for MMAShapeAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct MMATypesAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for MMATypesAttr<'c> {}

impl<'c> fmt::Display for MMATypesAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ReduxKindAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ReduxKindAttr<'c> {}

impl<'c> fmt::Display for ReduxKindAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ShflKindAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ShflKindAttr<'c> {}

impl<'c> fmt::Display for ShflKindAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
