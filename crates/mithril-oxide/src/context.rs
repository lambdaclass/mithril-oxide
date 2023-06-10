use mithril_oxide_sys as ffi;
use std::{cell::RefCell, mem::MaybeUninit};

// #[derive(Debug)]
pub struct Context {
    pub(crate) inner: RefCell<ffi::MLIRContext>,
}

impl Context {
    pub fn new(threaded: bool) -> Self {
        Self {
            inner: RefCell::new(unsafe {
                ffi::MLIRContext::new(if threaded {
                    ffi::Threading::ENABLED
                } else {
                    ffi::Threading::DISABLED
                })
            }),
        }
    }

    pub fn register_all_dialects(&mut self) {
        unsafe {
            ffi::registerAllDialects(self.inner.get_mut() as _);
        }
    }

    pub fn load_all_available_dialects(&mut self) {
        unsafe {
            self.inner.get_mut().loadAllAvailableDialects();
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new(true)
    }
}
