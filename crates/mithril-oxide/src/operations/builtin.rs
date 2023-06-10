use crate::{
    attributes::{builtin::StringAttr, LocationAttr},
    location::Location,
    util::{IntoWithContext, NotSet},
    Context,
};
use mithril_oxide_sys as ffi;
use std::{marker::PhantomData, ptr::null};

// TODO: Operation `builtin.module`.
pub struct ModuleOp<'c> {
    inner: ffi::ModuleOp,
    phantom: PhantomData<&'c Context>,
}

impl<'c> ModuleOp<'c> {
    pub fn builder(
        context: &'c Context,
    ) -> ModuleOpBuilder<&'static str, NotSet, StringAttr<'_>, StringAttr<'_>> {
        ModuleOpBuilder {
            context,
            name: None,
            location: NotSet,
            sym_name: None,
            sym_visibility: None,
        }
    }
}

#[doc(hidden)]
pub struct ModuleOpBuilder<'c, OpName, OpLocation, SymName, SymVisibility> {
    context: &'c Context,

    name: Option<OpName>,
    location: OpLocation,

    sym_name: Option<SymName>,
    sym_visibility: Option<SymVisibility>,
}

impl<'c, OpName, OpLocation, SymName, SymVisibility>
    ModuleOpBuilder<'c, OpName, OpLocation, SymName, SymVisibility>
{
    pub fn name<T>(self, value: T) -> ModuleOpBuilder<'c, T, OpLocation, SymName, SymVisibility> {
        ModuleOpBuilder {
            context: self.context,
            name: Some(value),
            location: self.location,
            sym_name: self.sym_name,
            sym_visibility: self.sym_visibility,
        }
    }

    pub fn location<T>(self, value: T) -> ModuleOpBuilder<'c, OpName, T, SymName, SymVisibility> {
        ModuleOpBuilder {
            context: self.context,
            name: self.name,
            location: value,
            sym_name: self.sym_name,
            sym_visibility: self.sym_visibility,
        }
    }

    pub fn sym_name<T>(
        self,
        value: T,
    ) -> ModuleOpBuilder<'c, OpName, OpLocation, T, SymVisibility> {
        ModuleOpBuilder {
            context: self.context,
            name: self.name,
            location: self.location,
            sym_name: Some(value),
            sym_visibility: self.sym_visibility,
        }
    }

    pub fn sym_visibility<T>(
        self,
        value: T,
    ) -> ModuleOpBuilder<'c, OpName, OpLocation, SymName, T> {
        ModuleOpBuilder {
            context: self.context,
            name: self.name,
            location: self.location,
            sym_name: self.sym_name,
            sym_visibility: Some(value),
        }
    }

    pub fn build(self) -> ModuleOp<'c>
    where
        OpName: AsRef<str> + std::fmt::Debug,
        OpLocation: LocationAttr<'c>,
        SymName: IntoWithContext<StringAttr<'c>>,
        SymVisibility: IntoWithContext<StringAttr<'c>>,
    {
        let name = dbg!(self.name).map(|value| {
            let value = value.as_ref();
            unsafe { ffi::StringRef::new(value.as_ptr() as _, value.len() as u64) }
        });
        let location: self::Location<'c> = self.location.into();

        let mut ffi_module = unsafe {
            ffi::_ModuleOp_create(location.inner, name.map(|x| &x as _).unwrap_or(null()))
        };

        if let Some(sym_name) = self.sym_name {
            let sym_name = sym_name.into_with_context(self.context);
            unsafe {
                ffi_module.setSymNameAttr(sym_name.inner);
            }
        }
        if let Some(sym_visibility) = self.sym_visibility {
            let sym_visibility = sym_visibility.into_with_context(self.context);
            unsafe {
                ffi_module.setSymVisibilityAttr(sym_visibility.inner);
            }
        }

        ModuleOp {
            inner: ffi_module,
            phantom: PhantomData,
        }
    }
}

// TODO: Operation `builtin.unrealized_conversion_cast`.
