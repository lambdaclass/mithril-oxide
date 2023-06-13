use mithril_oxide_sys as ffi;
use std::{cell::RefCell, mem::MaybeUninit};

/// The MLIR context.
#[derive(Debug)]
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

    /// Loads all available dialects.
    pub fn load_all_available_dialects(&mut self) {
        unsafe {
            self.inner.get_mut().pin_mut().loadAllAvailableDialects();
        }
    }

    /// Return whether unregistered dialects are allowed.
    pub fn allows_unregistered_dialects(&self) -> bool {
        self.inner
            .borrow_mut()
            .pin_mut()
            .allowsUnregisteredDialects()
    }

    /// Set whether to allow unregistered dialects.
    pub fn allow_unregistered_dialects(&mut self, allow: bool) {
        self.inner
            .borrow_mut()
            .pin_mut()
            .allowUnregisteredDialects(allow);
    }

    /// Set whether multithreading is enabled for this context.
    pub fn enable_multithreading(&mut self, enable: bool) {
        self.inner
            .borrow_mut()
            .pin_mut()
            .enableMultithreading(enable);
    }

    /// Get if multithreading is enabled.
    pub fn is_multithreading_enabled(&self) -> bool {
        self.inner.borrow_mut().pin_mut().isMultithreadingEnabled()
    }
}

impl Default for Context {
    /// Create the context with default parameters. Threading = on.
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let context = Context::new();
        assert!(!context.inner.borrow().is_null());
        assert!(context.is_multithreading_enabled());
    }
}
