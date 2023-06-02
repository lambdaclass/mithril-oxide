use std::fmt::Display;

mod acc;
mod affine;
mod amdgpu;
mod amx;
mod arith;
mod arm_neon;
mod arm_sve;
mod r#async;
mod bufferization;
mod builtin;
mod cf;
mod complex;
mod emitc;
mod func;
mod gpu;
mod index;
mod linalg;
mod llvm;
mod math;
mod memref;
mod ml_program;
mod nvgpu;
mod nvvm;
mod omp;
mod pdl;
mod pdl_interp;
mod quant;
mod rocdl;
mod scf;
mod shape;
mod sparse_tensor;
mod spirv;
mod tensor;
mod tosa;
mod transform;
mod vector;
mod x86vector;

pub trait Operation
where
    Self: Display,
{
}
