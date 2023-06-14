use super::Operation;
use crate::{
    attributes::{builtin::StringAttr, LocationAttr},
    location::Location,
    util::{FromWithContext, NotSet},
    Context, Region,
};
use mithril_oxide_sys as ffi;
use std::{fmt, marker::PhantomData, ptr::null};

// TODO: Operation `builtin.module`.
pub struct ModuleOp<'c> {
    inner: ffi::UniquePtr<ffi::IR::BuiltinOps::ModuleOp>,
    phantom: PhantomData<&'c Context>,
}

impl<'c> ModuleOp<'c> {
    pub fn builder() -> ModuleOpBuilder<'c, NotSet, StringAttr<'c>, StringAttr<'c>> {
        ModuleOpBuilder {
            phantom: PhantomData,
            location: NotSet,
            sym_name: None,
            sym_visibility: None,
        }
    }

    pub fn body_mut(&mut self) -> &mut Region {
        Region::from_ffi(self.inner.pin_mut().getBodyRegion())
    }
}

impl<'c> Operation for ModuleOp<'c> {
    fn num_results(&self) -> usize {
        todo!()
    }

    fn result(&self, index: usize) -> super::OperationResult {
        todo!()
    }
}

impl<'c> fmt::Display for ModuleOp<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[doc(hidden)]
pub struct ModuleOpBuilder<'c, OpLocation, SymName, SymVisibility> {
    phantom: PhantomData<&'c Context>,
    location: OpLocation,

    sym_name: Option<SymName>,
    sym_visibility: Option<SymVisibility>,
}

impl<'c, OpLocation, SymName, SymVisibility>
    ModuleOpBuilder<'c, OpLocation, SymName, SymVisibility>
{
    pub fn location<T>(self, value: T) -> ModuleOpBuilder<'c, T, SymName, SymVisibility> {
        ModuleOpBuilder {
            phantom: self.phantom,
            location: value,
            sym_name: self.sym_name,
            sym_visibility: self.sym_visibility,
        }
    }

    pub fn sym_name<T>(self, value: T) -> ModuleOpBuilder<'c, OpLocation, T, SymVisibility> {
        ModuleOpBuilder {
            phantom: self.phantom,
            location: self.location,
            sym_name: Some(value),
            sym_visibility: self.sym_visibility,
        }
    }

    pub fn sym_visibility<T>(self, value: T) -> ModuleOpBuilder<'c, OpLocation, SymName, T> {
        ModuleOpBuilder {
            phantom: self.phantom,
            location: self.location,
            sym_name: self.sym_name,
            sym_visibility: Some(value),
        }
    }

    /// `ModuleOp` is a special top-level operation, therefore it has a public `build()` method.
    /// Normal operations would pass the builder to `Block::push()`.
    pub fn build(self, context: &'c Context) -> ModuleOp<'c>
    where
        OpLocation: LocationAttr<'c>,
        StringAttr<'c>: FromWithContext<SymName>,
        StringAttr<'c>: FromWithContext<SymVisibility>,
    {
        let location: self::Location<'c> = self.location.into();

        let mut ffi_module = ffi::IR::BuiltinOps::ModuleOp::new(&location.inner);

        if let Some(sym_name) = self.sym_name {
            let sym_name = StringAttr::from_with_context(sym_name, context);
            unsafe {
                ffi_module.pin_mut().setSymNameAttr(&sym_name.inner);
            }
        }
        if let Some(sym_visibility) = self.sym_visibility {
            let sym_visibility = StringAttr::from_with_context(sym_visibility, context);
            unsafe {
                ffi_module
                    .pin_mut()
                    .setSymVisibilityAttr(&sym_visibility.inner);
            }
        }

        ModuleOp {
            inner: ffi_module,
            phantom: PhantomData,
        }
    }
}

// TODO: Operation `builtin.unrealized_conversion_cast`.
