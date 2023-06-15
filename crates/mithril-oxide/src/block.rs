use crate::{attributes::LocationAttr, operations::OperationBuilder, types::Type, Context};
use mithril_oxide_sys as ffi;
use std::{marker::PhantomData, pin::Pin};

#[repr(transparent)]
pub struct Block<'c> {
    inner: *mut ffi::IR::Block::Block,
    phantom: PhantomData<&'c Context>,
}

impl<'c> Block<'c> {
    pub(crate) fn from_ffi(inner: Pin<&mut ffi::IR::Block::Block>) -> &mut Self {
        // This is guaranteed to be safe by `#[repr(transparent)]`.
        unsafe { std::mem::transmute(inner) }
    }

    pub fn add_argument(&mut self, location: impl LocationAttr<'c>, r#type: impl AsRef<Type<'c>>) {
        todo!()
    }

    pub fn push<Op>(&mut self, builder: impl OperationBuilder<'c, Target = Op>) -> &Op {
        // let op = builder.build();
        todo!()
    }
}
