use mithril_oxide_sys as ffi;
use std::{cell::RefCell, mem::MaybeUninit};

// #[derive(Debug)]
pub struct Context {
    pub(crate) inner: RefCell<ffi::UniquePtr<ffi::IR::MLIRContext::MLIRContext>>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(ffi::IR::MLIRContext::MLIRContext::new()),
        }
    }

    pub fn register_all_dialects(&mut self) {
        unsafe {
            ffi::InitAllDialects::registerAllDialects(self.inner.get_mut().pin_mut());
        }
    }

    pub fn load_all_available_dialects(&mut self) {
        unsafe {
            self.inner.get_mut().pin_mut().loadAllAvailableDialects();
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
