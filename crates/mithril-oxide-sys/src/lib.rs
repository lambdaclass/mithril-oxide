#![allow(non_snake_case)]
#![deny(clippy::pedantic)]
#![deny(warnings)]
//
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_safety_doc)] // todo: remove
#![allow(clippy::doc_markdown)] // todo: remove

pub use cxx::UniquePtr;

pub mod Dialect;
pub mod IR;
pub mod InitAllDialects;
pub mod InitAllTranslations;
