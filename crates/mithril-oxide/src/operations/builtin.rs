use crate::{
    attributes::{builtin::StringAttr, LocationAttr},
    location::Location,
    util::{IntoWithContext, NotSet},
    Context,
};
use mithril_oxide_cxx as ffi;
use std::{marker::PhantomData, ptr::null};

// TODO: Operation `builtin.module`.
pub struct ModuleOp<'c> {
    inner: ffi::UniquePtr<ffi::IR::BuiltinOps::ModuleOp>,
    phantom: PhantomData<&'c Context>,
}

impl<'c> ModuleOp<'c> {
    pub fn builder(
        context: &'c Context,
    ) -> ModuleOpBuilder<NotSet, StringAttr<'_>, StringAttr<'_>> {
        ModuleOpBuilder {
            context,
            location: NotSet,
            sym_name: None,
            sym_visibility: None,
        }
    }
}

#[doc(hidden)]
pub struct ModuleOpBuilder<'c, OpLocation, SymName, SymVisibility> {
    context: &'c Context,
    location: OpLocation,

    sym_name: Option<SymName>,
    sym_visibility: Option<SymVisibility>,
}

impl<'c, OpLocation, SymName, SymVisibility>
    ModuleOpBuilder<'c, OpLocation, SymName, SymVisibility>
{
    pub fn location<T>(self, value: T) -> ModuleOpBuilder<'c, T, SymName, SymVisibility> {
        ModuleOpBuilder {
            context: self.context,
            location: value,
            sym_name: self.sym_name,
            sym_visibility: self.sym_visibility,
        }
    }

    pub fn sym_name<T>(self, value: T) -> ModuleOpBuilder<'c, OpLocation, T, SymVisibility> {
        ModuleOpBuilder {
            context: self.context,
            location: self.location,
            sym_name: Some(value),
            sym_visibility: self.sym_visibility,
        }
    }

    pub fn sym_visibility<T>(self, value: T) -> ModuleOpBuilder<'c, OpLocation, SymName, T> {
        ModuleOpBuilder {
            context: self.context,
            location: self.location,
            sym_name: self.sym_name,
            sym_visibility: Some(value),
        }
    }

    pub fn build(self) -> ModuleOp<'c>
    where
        OpLocation: LocationAttr<'c>,
        SymName: IntoWithContext<StringAttr<'c>>,
        SymVisibility: IntoWithContext<StringAttr<'c>>,
    {
        let location: self::Location<'c> = self.location.into();

        let mut ffi_module = ffi::IR::BuiltinOps::ModuleOp::new(&location.inner);

        if let Some(sym_name) = self.sym_name {
            let sym_name = sym_name.into_with_context(self.context);
            unsafe {
                ffi_module.pin_mut().setSymNameAttr(sym_name.inner);
            }
        }
        if let Some(sym_visibility) = self.sym_visibility {
            let sym_visibility = sym_visibility.into_with_context(self.context);
            unsafe {
                ffi_module
                    .pin_mut()
                    .setSymVisibilityAttr(sym_visibility.inner);
            }
        }

        ModuleOp {
            inner: ffi_module,
            phantom: PhantomData,
        }
    }
}

// TODO: Operation `builtin.unrealized_conversion_cast`.
