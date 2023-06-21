use crate::Context;
use mithril_oxide_sys as ffi;
use std::{
    fmt::{self, Display},
    iter::{empty, Empty},
    marker::PhantomData,
};

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

macro_rules! impl_type {
    ( $name:ident $({ $attr:ident : $ty:path })? ) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub struct $name<'c>($crate::types::Type<'c>);

        impl<'c> AsRef<$crate::types::Type<'c>> for $name<'c> {
            fn as_ref(&self) -> &$crate::types::Type<'c> {
                &self.0
            }
        }

        // impl<'c> fmt::Display for $name<'c> {
        //     todo!()
        // }
    };
}
pub(self) use impl_type;

macro_rules! impl_type_new {
    ( $name:ident $({ $( $( $arg_name:ident : $arg_ty:ty ),+ $(,)? )? })? ) => {
        impl<'c> $name<'c> {
            #[allow(clippy::new_without_default)]
            pub fn new($($($($arg_name: $arg_ty),+)?)?) -> Self {
                // concat_idents!($name, _new)();
                todo!()
            }
        }
    };
}
pub(self) use impl_type_new;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Type<'c> {
    pub(crate) ptr: *mut std::ffi::c_void,
    phantom: std::marker::PhantomData<&'c Context>,
}

impl<'c> fmt::Display for Type<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

// pub trait Type<'c>
// where
//     Self: Clone + Display,
// {
//     fn size(&self) -> usize;
//     fn size_in_bits(&self) -> usize;

//     fn abi_alignment(&self) -> usize;
//     fn preferred_alignment(&self) -> usize;
// }

// pub struct DynType<'c> {
//     inner: ffi::IR::Types::Type,
//     phantom: PhantomData<&'c Context>,
// }

// pub trait TypeTuple<'c> {
//     type Iterator: Iterator<Item = DynType<'c>>;

//     fn into_iter(self) -> Self::Iterator;
// }

// impl<'c> TypeTuple<'c> for () {
//     type Iterator = std::array::IntoIter<DynType<'c>, 0>;

//     fn into_iter(self) -> Self::Iterator {
//         [].into_iter()
//     }
// }

// impl<'c, T0> TypeTuple<'c> for (T0,) {
//     type Iterator = std::array::IntoIter<DynType<'c>, 1>;

//     fn into_iter(self) -> Self::Iterator {
//         [self.0.into()].into_iter()
//     }
// }

// impl<'c, T0, T1> TypeTuple<'c> for (T0, T1) {
//     type Iterator = std::array::IntoIter<DynType<'c>, 2>;

//     fn into_iter(self) -> Self::Iterator {
//         [self.0.into(), self.1.into()].into_iter()
//     }
// }

// impl<'c, T0, T1, T2> TypeTuple<'c> for (T0, T1, T2) {
//     type Iterator = std::array::IntoIter<DynType<'c>, 3>;

//     fn into_iter(self) -> Self::Iterator {
//         [self.0.into(), self.1.into(), self.2.into()].into_iter()
//     }
// }

// impl<'c, T0, T1, T2, T3> TypeTuple<'c> for (T0, T1, T2, T3) {
//     type Iterator = std::array::IntoIter<DynType<'c>, 4>;

//     fn into_iter(self) -> Self::Iterator {
//         [self.0.into(), self.1.into(), self.2.into(), self.3.into()].into_iter()
//     }
// }
