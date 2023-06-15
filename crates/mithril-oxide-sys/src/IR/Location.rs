pub use self::ffi::{
    CallSiteLoc, FileLineColLoc, FusedLoc, Location, NameLoc, OpaqueLoc, UnknownLoc,
};
use crate::IR::MLIRContext::MLIRContext;
use cxx::UniquePtr;
use std::{fmt, pin::Pin};

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Location.hpp");

        type CallSiteLoc;
        type FileLineColLoc;
        type FusedLoc;
        type Location;
        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
        type NameLoc;
        type OpaqueLoc;
        type UnknownLoc;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/IR/Location.hpp");

        #[must_use]
        fn UnknownLoc_get(context: Pin<&mut MLIRContext>) -> UniquePtr<UnknownLoc>;
        #[must_use]
        fn UnknownLoc_to_Location(loc: &UnknownLoc) -> UniquePtr<Location>;
    }
}

impl fmt::Debug for ffi::Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Location").finish_non_exhaustive()
    }
}

impl ffi::UnknownLoc {
    #[must_use]
    pub fn get(context: Pin<&mut MLIRContext>) -> UniquePtr<Self> {
        ffi::UnknownLoc_get(context)
    }
}

impl fmt::Debug for ffi::UnknownLoc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("UnknownLoc").finish_non_exhaustive()
    }
}

impl From<&ffi::UnknownLoc> for UniquePtr<ffi::Location> {
    fn from(val: &ffi::UnknownLoc) -> Self {
        ffi::UnknownLoc_to_Location(val)
    }
}
