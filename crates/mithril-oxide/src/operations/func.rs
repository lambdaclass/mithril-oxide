use crate::{util::NotSet, Context};
use mithril_oxide_sys as ffi;
use std::marker::PhantomData;

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
    pub fn builder(
        context: &'c Context,
    ) -> FuncOpBuilder<'c, NotSet, NotSet, NotSet, NotSet, NotSet, NotSet> {
        FuncOpBuilder {
            context,
            location: NotSet,
            sym_name: NotSet,
            function_type: NotSet,
            sym_visibility: NotSet,
            arg_attrs: NotSet,
            res_attrs: NotSet,
        }
    }
}

#[doc(hidden)]
pub struct FuncOpBuilder<'c, OpLocation, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs> {
    context: &'c Context,
    location: OpLocation,

    sym_name: SymName,
    function_type: FunctionType,
    sym_visibility: SymVisibility,
    arg_attrs: ArgAttrs,
    res_attrs: ResAttrs,
}

impl<'c, OpLocation, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs>
    FuncOpBuilder<'c, OpLocation, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs>
{
    pub fn location<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, T, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs> {
        todo!()
    }

    pub fn sym_name<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, T, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs> {
        todo!()
    }

    pub fn function_type<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, T, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs> {
        todo!()
    }

    pub fn sym_visibility<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, T, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs> {
        todo!()
    }

    pub fn arg_attrs<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, T, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs> {
        todo!()
    }

    pub fn res_attrs<T>(
        self,
        value: T,
    ) -> FuncOpBuilder<'c, T, SymName, FunctionType, SymVisibility, ArgAttrs, ResAttrs> {
        todo!()
    }
}
