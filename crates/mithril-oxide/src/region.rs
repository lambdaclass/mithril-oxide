use crate::{Block, Context};
use mithril_oxide_sys as ffi;
use std::{marker::PhantomData, pin::Pin};

#[repr(transparent)]
pub struct Region<'c> {
    inner: *mut ffi::IR::Region::Region,
    phantom: PhantomData<&'c Context>,
}

impl<'c> Region<'c> {
    pub(crate) fn from_ffi(inner: Pin<&mut ffi::IR::Region::Region>) -> &mut Self {
        // This is guaranteed to be safe by `#[repr(transparent)]`.
        unsafe { std::mem::transmute(inner) }
    }

    pub fn emplace_block(&mut self) -> &mut Block {
        let inner: Pin<&mut ffi::IR::Region::Region> =
            unsafe { std::mem::transmute(&mut *self.inner) };
        Block::from_ffi(inner.emplaceBlock())
    }
}
