use super::Attribute;
use crate::{context::Context, util::FromWithContext};
use mithril_oxide_sys as ffi;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct AffineMapAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for AffineMapAttr<'c> {}

impl<'c> fmt::Display for AffineMapAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ArrayAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ArrayAttr<'c> {}

impl<'c> fmt::Display for ArrayAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct CallSiteLoc<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for CallSiteLoc<'c> {}

impl<'c> fmt::Display for CallSiteLoc<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct DenseArrayAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for DenseArrayAttr<'c> {}

impl<'c> fmt::Display for DenseArrayAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct DenseIntOrFPElementsAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for DenseIntOrFPElementsAttr<'c> {}

impl<'c> fmt::Display for DenseIntOrFPElementsAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct DenseResourceElementsAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for DenseResourceElementsAttr<'c> {}

impl<'c> fmt::Display for DenseResourceElementsAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct DenseStringElementsAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for DenseStringElementsAttr<'c> {}

impl<'c> fmt::Display for DenseStringElementsAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct DictionaryAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for DictionaryAttr<'c> {}

impl<'c> fmt::Display for DictionaryAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct FileLineColLoc<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for FileLineColLoc<'c> {}

impl<'c> fmt::Display for FileLineColLoc<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct FloatAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for FloatAttr<'c> {}

impl<'c> fmt::Display for FloatAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct FusedLoc<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for FusedLoc<'c> {}

impl<'c> fmt::Display for FusedLoc<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct IntegerAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for IntegerAttr<'c> {}

impl<'c> fmt::Display for IntegerAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct IntegerSetAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for IntegerSetAttr<'c> {}

impl<'c> fmt::Display for IntegerSetAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct NameLoc<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for NameLoc<'c> {}

impl<'c> fmt::Display for NameLoc<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

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

#[derive(Debug)]
pub struct OpaqueLoc<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for OpaqueLoc<'c> {}

impl<'c> fmt::Display for OpaqueLoc<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct SparseElementsAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for SparseElementsAttr<'c> {}

impl<'c> fmt::Display for SparseElementsAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct StridedLayoutAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for StridedLayoutAttr<'c> {}

impl<'c> fmt::Display for StridedLayoutAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

// #[derive(Debug)]
pub struct StringAttr<'c> {
    pub(crate) inner: ffi::UniquePtr<ffi::IR::BuiltinAttributes::StringAttr>,
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for StringAttr<'c> {}

impl<'c> fmt::Display for StringAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

impl<'c> FromWithContext<&str> for StringAttr<'c> {
    fn from_with_context(value: &str, context: &Context) -> Self {
        todo!()
    }
}

#[derive(Debug)]
pub struct SymbolRefAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for SymbolRefAttr<'c> {}

impl<'c> fmt::Display for SymbolRefAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct TypeAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for TypeAttr<'c> {}

impl<'c> fmt::Display for TypeAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct UnitAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for UnitAttr<'c> {}

impl<'c> fmt::Display for UnitAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

// #[derive(Debug)]
pub struct UnknownLoc<'c> {
    pub(crate) inner: ffi::UniquePtr<ffi::IR::Location::UnknownLoc>,
    phantom: PhantomData<&'c Context>,
}

impl<'c> UnknownLoc<'c> {
    pub fn new(context: &'c Context) -> Self {
        Self {
            inner: unsafe {
                ffi::IR::Location::UnknownLoc::get(context.inner.borrow_mut().pin_mut())
            },
            phantom: PhantomData,
        }
    }
}

impl<'c> Attribute for UnknownLoc<'c> {}

impl<'c> fmt::Display for UnknownLoc<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
