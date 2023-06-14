use crate::Context;
use mithril_oxide_sys as ffi;
use std::{
    fmt::{self, Display},
    marker::PhantomData,
    pin::{pin, Pin},
};

pub mod acc;
pub mod affine;
pub mod amdgpu;
pub mod amx;
pub mod arith;
pub mod arm_neon;
pub mod arm_sve;
pub mod r#async;
pub mod bufferization;
pub mod builtin;
pub mod cf;
pub mod complex;
pub mod emitc;
pub mod func;
pub mod gpu;
pub mod index;
pub mod linalg;
pub mod llvm;
pub mod math;
pub mod memref;
pub mod ml_program;
pub mod nvgpu;
pub mod nvvm;
pub mod omp;
pub mod pdl;
pub mod pdl_interp;
pub mod quant;
pub mod rocdl;
pub mod scf;
pub mod shape;
pub mod sparse_tensor;
pub mod spirv;
pub mod tensor;
pub mod tosa;
pub mod transform;
pub mod vector;
pub mod x86vector;

pub trait Operation
where
    Self: Display,
{
    fn num_results(&self) -> usize;
    fn result(&self, index: usize) -> OperationResult;
}

pub trait OperationBuilder<'c> {
    type Target: Operation;

    fn build(self, context: &'c Context) -> Self::Target;
}

pub struct OperationResult<'a> {
    // inner: !,
    phantom: PhantomData<&'a dyn Operation>,
}

pub struct DynOperation<'c> {
    inner: *mut ffi::IR::Operation::Operation,
    phanom: PhantomData<&'c Context>,
}

impl<'c> DynOperation<'c> {
    pub fn builder(name: &str) -> DynOperationBuilder<'c> {
        todo!()
    }
}

impl<'c> Operation for DynOperation<'c> {
    fn num_results(&self) -> usize {
        todo!()
    }

    fn result(&self, index: usize) -> OperationResult {
        todo!()
    }
}

impl<'c> fmt::Display for DynOperation<'c> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

pub struct DynOperationBuilder<'c> {
    context: &'c Context,
}

impl<'c> OperationBuilder<'c> for DynOperationBuilder<'c> {
    type Target = DynOperation<'c>;

    fn build(self, context: &'c Context) -> Self::Target {
        todo!()
    }
}
