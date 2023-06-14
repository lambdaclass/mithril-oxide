#![allow(clippy::module_name_repetitions)]
#![allow(non_snake_case)]
#![deny(clippy::pedantic)]
#![deny(warnings)]

pub use cxx::UniquePtr;

pub mod Dialect;
pub mod IR;
pub mod InitAllDialects;
pub mod InitAllTranslations;
