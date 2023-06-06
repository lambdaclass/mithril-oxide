use super::Attribute;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct SparseTensorDimSliceAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for SparseTensorDimSliceAttr<'c> {}

impl<'c> fmt::Display for SparseTensorDimSliceAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct SparseTensorEncodingAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for SparseTensorEncodingAttr<'c> {}

impl<'c> fmt::Display for SparseTensorEncodingAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct SparseTensorSortKindAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for SparseTensorSortKindAttr<'c> {}

impl<'c> fmt::Display for SparseTensorSortKindAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct StorageSpecifierKindAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for StorageSpecifierKindAttr<'c> {}

impl<'c> fmt::Display for StorageSpecifierKindAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
