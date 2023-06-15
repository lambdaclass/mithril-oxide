use crate::operations::Operation;
use mithril_oxide_sys as ffi;
use std::marker::PhantomData;

pub trait Value<'c> {}

pub struct OperationResult<'c, 'a> {
    inner: ffi::UniquePtr<ffi::IR::Operation::OpResult>,
    phantom: PhantomData<&'a dyn Operation<'c>>,
}

impl<'c, 'a> Value<'c> for OperationResult<'c, 'a> {}
