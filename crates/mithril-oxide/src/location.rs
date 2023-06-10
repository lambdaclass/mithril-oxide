pub use crate::attributes::builtin::{
    CallSiteLoc, FileLineColLoc, FusedLoc, NameLoc, OpaqueLoc, UnknownLoc,
};
use crate::Context;
use mithril_oxide_sys as ffi;
use std::marker::PhantomData;

/// A generic location.
pub struct Location<'c> {
    pub(crate) inner: ffi::Location,
    phantom: PhantomData<&'c Context>,
}

impl<'c> From<CallSiteLoc<'c>> for Location<'c> {
    fn from(value: CallSiteLoc<'c>) -> Self {
        todo!()
    }
}

impl<'c> From<FileLineColLoc<'c>> for Location<'c> {
    fn from(value: FileLineColLoc<'c>) -> Self {
        todo!()
    }
}

impl<'c> From<FusedLoc<'c>> for Location<'c> {
    fn from(value: FusedLoc<'c>) -> Self {
        todo!()
    }
}

impl<'c> From<NameLoc<'c>> for Location<'c> {
    fn from(value: NameLoc<'c>) -> Self {
        todo!()
    }
}

impl<'c> From<OpaqueLoc<'c>> for Location<'c> {
    fn from(value: OpaqueLoc<'c>) -> Self {
        todo!()
    }
}

impl<'c> From<UnknownLoc<'c>> for Location<'c> {
    fn from(value: UnknownLoc<'c>) -> Self {
        Self {
            inner: unsafe { ffi::_UnknownLoc_downgradeTo_Location(value.inner) },
            phantom: PhantomData,
        }
    }
}
