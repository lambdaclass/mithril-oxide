use super::{Operation, OperationBuilder};
use crate::{
    attributes::{
        builtin::{DictionaryAttr, StringAttr},
        LocationAttr,
    },
    types::builtin::FunctionType,
    util::{FromWithContext, NotSet},
    Context, Region,
};
use mithril_oxide_sys as ffi;
use std::{fmt, marker::PhantomData};

// TODO: Operation `func.call_indirect`.

// TODO: Operation `func.call`.

// TODO: Operation `func.constant`.

// TODO: Operation `func.func`.

// TODO: Operation `func.return`.
pub struct FuncOp<'c> {
    inner: ffi::UniquePtr<ffi::Dialect::Func::IR::FuncOps::FuncOp>,
    phantom: PhantomData<&'c Context>,
}

impl<'c> FuncOp<'c> {
    pub fn builder() -> FuncOpBuilder<
        'c,
        NotSet,
        NotSet,
        NotSet,
        StringAttr<'c>,
        DictionaryAttr<'c>,
        DictionaryAttr<'c>,
    > {
        FuncOpBuilder {
            phantom: PhantomData,
            location: NotSet,
            sym_name: NotSet,
            function_type: NotSet,
            sym_visibility: None,
            arg_attrs: None,
            res_attrs: None,
        }
    }

    pub fn body_mut(&mut self) -> &mut Region {
        // Region::from_ffi(self.inner.pin_mut().getBodyRegion())
        todo!()
    }
}

impl<'c> Operation<'c> for FuncOp<'c> {
    fn num_results(&self) -> usize {
        todo!()
    }

    fn result(&self, index: usize) -> super::OperationResult<'c, '_> {
        todo!()
    }
}

impl<'c> fmt::Display for FuncOp<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[doc(hidden)]
pub struct FuncOpBuilder<'c, OpLocation, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs> {
    phantom: PhantomData<&'c Context>,
    location: OpLocation,

    sym_name: SymName,
    function_type: FunctionType,
    sym_visibility: Option<SymVisibility>,
    arg_attrs: Option<ArgAttrs>,
    res_attrs: Option<ResAttrs>,
}

impl<'c, OpLocation, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs>
    FuncOpBuilder<'c, OpLocation, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs>
{
    pub fn location<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, T, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs> {
        FuncOpBuilder {
            phantom: self.phantom,
            location: value,
            sym_name: self.sym_name,
            function_type: self.function_type,
            sym_visibility: self.sym_visibility,
            arg_attrs: self.arg_attrs,
            res_attrs: self.res_attrs,
        }
    }

    pub fn sym_name<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, OpLocation, T, FunctionType, SymVisibility, ArgAttrs, ResAttrs> {
        FuncOpBuilder {
            phantom: self.phantom,
            location: self.location,
            sym_name: value,
            function_type: self.function_type,
            sym_visibility: self.sym_visibility,
            arg_attrs: self.arg_attrs,
            res_attrs: self.res_attrs,
        }
    }

    pub fn function_type<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, OpLocation, SymName, T, SymVisibility, ArgAttrs, ResAttrs> {
        FuncOpBuilder {
            phantom: self.phantom,
            location: self.location,
            sym_name: self.sym_name,
            function_type: value,
            sym_visibility: self.sym_visibility,
            arg_attrs: self.arg_attrs,
            res_attrs: self.res_attrs,
        }
    }

    pub fn sym_visibility<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, OpLocation, SymName, FunctionType, T, ArgAttrs, ResAttrs> {
        FuncOpBuilder {
            phantom: self.phantom,
            location: self.location,
            sym_name: self.sym_name,
            function_type: self.function_type,
            sym_visibility: Some(value),
            arg_attrs: self.arg_attrs,
            res_attrs: self.res_attrs,
        }
    }

    pub fn arg_attrs<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, OpLocation, SymName, FunctionType, SymVisibility, T, ResAttrs> {
        FuncOpBuilder {
            phantom: self.phantom,
            location: self.location,
            sym_name: self.sym_name,
            function_type: self.function_type,
            sym_visibility: self.sym_visibility,
            arg_attrs: Some(value),
            res_attrs: self.res_attrs,
        }
    }

    pub fn res_attrs<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, OpLocation, SymName, FunctionType, SymVisibility, ArgAttrs, T> {
        FuncOpBuilder {
            phantom: self.phantom,
            location: self.location,
            sym_name: self.sym_name,
            function_type: self.function_type,
            sym_visibility: self.sym_visibility,
            arg_attrs: self.arg_attrs,
            res_attrs: Some(value),
        }
    }
}

impl<'c, OpLocation, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs> OperationBuilder<'c>
    for FuncOpBuilder<'c, OpLocation, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs>
where
    OpLocation: LocationAttr<'c>,
    StringAttr<'c>: FromWithContext<SymName>,
    self::FunctionType<'c>: FromWithContext<FunctionType>,
    StringAttr<'c>: FromWithContext<SymVisibility>,
    // : IntoWithContext<ArgAttrs>,
    // : IntoWithContext<ResAttrs>,
{
    type Target = FuncOp<'c>;

    fn build(self, context: &'c Context) -> Self::Target {
        todo!()
    }
}
