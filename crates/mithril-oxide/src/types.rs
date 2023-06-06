use std::fmt::Display;

pub mod acc;
pub mod r#async;
pub mod builtin;
pub mod emitc;
pub mod llvm;
pub mod ml_program;
pub mod nvgpu;
pub mod pdl;
pub mod shape;
pub mod sparse_tensor;
pub mod spirv;
pub mod transform;

pub trait Type
where
    Self: Display,
{
    fn size(&self) -> usize;
    fn size_in_bits(&self) -> usize;

    fn abi_alignment(&self) -> usize;
    fn preferred_alignment(&self) -> usize;
}
