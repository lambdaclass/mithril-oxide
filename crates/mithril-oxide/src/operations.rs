use crate::{value::OperationResult, Context};
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

pub trait Operation<'c>
where
    Self: Display,
{
    fn num_results(&self) -> usize;
    fn result(&self, index: usize) -> OperationResult<'c, '_>;
}

pub trait OperationBuilder<'c> {
    type Target: Operation<'c>;

    fn build(self, context: &'c Context) -> Self::Target;
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

impl<'c> Operation<'c> for DynOperation<'c> {
    fn num_results(&self) -> usize {
        let op = unsafe { Pin::new_unchecked(&mut *self.inner) };
        op.getNumResults() as usize
    }

    fn result(&self, index: usize) -> OperationResult<'c, '_> {
        let op = unsafe { Pin::new_unchecked(&mut *self.inner) };
        if index >= self.num_results() {
            panic!("index out of bounds");
        }
        let result = op.getResult(index.try_into().unwrap());
        let res = unsafe { OperationResult::from_ffi(result.cast()) };
        res
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
