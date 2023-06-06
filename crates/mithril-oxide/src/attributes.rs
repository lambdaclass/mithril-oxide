use std::fmt::Display;

pub mod acc;
pub mod amdgpu;
pub mod arith;
pub mod builtin;
pub mod emitc;
pub mod index;
pub mod ml_program;
pub mod nvvm;
pub mod omp;
pub mod sparse_tensor;

pub trait Attribute
where
    Self: Display,
{
}
