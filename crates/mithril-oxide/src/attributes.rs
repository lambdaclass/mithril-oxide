use crate::location::Location;
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

/// Marker implemented by location attributes.
pub trait LocationAttr<'c>
where
    Self: Attribute + Into<Location<'c>>,
{
}

impl<'c> LocationAttr<'c> for self::builtin::CallSiteLoc<'c> {}
impl<'c> LocationAttr<'c> for self::builtin::FileLineColLoc<'c> {}
impl<'c> LocationAttr<'c> for self::builtin::FusedLoc<'c> {}
impl<'c> LocationAttr<'c> for self::builtin::NameLoc<'c> {}
impl<'c> LocationAttr<'c> for self::builtin::OpaqueLoc<'c> {}
impl<'c> LocationAttr<'c> for self::builtin::UnknownLoc<'c> {}
