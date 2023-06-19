#![allow(non_snake_case)]
#![deny(clippy::pedantic)]
#![deny(warnings)]
// The following rules must be kept after `deny(clippy::pedantic)` for them to be effective.
#![allow(clippy::doc_markdown)] // todo: remove
#![allow(clippy::missing_safety_doc)] // todo: remove
#![allow(clippy::module_name_repetitions)]

pub(crate) use self::ffi::c_void;
pub use cxx::UniquePtr;

pub mod Dialect;
pub mod IR;
pub mod InitAllDialects;
pub mod InitAllTranslations;
pub mod Interfaces;

#[cxx::bridge]
mod ffi {
    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/lib.hpp");

        type c_void;
    }
}
