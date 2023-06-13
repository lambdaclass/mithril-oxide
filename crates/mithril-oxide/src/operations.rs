use std::{
    fmt::Display,
    pin::{pin, Pin},
};

use mithril_oxide_sys::{UniquePtr, IR};

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

pub(crate) mod operation_impl {
    use super::*;

    pub trait OperationInner {
        fn get_inner(&self) -> &IR::Operation::Operation;
        fn get_inner_mut(&self) -> Pin<&mut IR::Operation::Operation>;
    }
}

pub trait Operation: operation_impl::OperationInner
where
    Self: Display,
{
    fn get_name(&self) -> &str {
        self.get_inner_mut().get_name()
    }
}
