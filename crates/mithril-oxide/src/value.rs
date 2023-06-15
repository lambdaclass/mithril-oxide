use crate::operations::Operation;
use mithril_oxide_sys as ffi;
use std::{ffi::c_void, marker::PhantomData};

pub trait Value<'c> {}

pub struct OperationResult<'c, 'a> {
    inner: *mut c_void,
    phantom: PhantomData<&'a dyn Operation<'c>>,
}

impl<'c, 'a> Value<'c> for OperationResult<'c, 'a> {}
