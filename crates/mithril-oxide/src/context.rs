use std::cell::RefCell;

use mithril_oxide_sys::{UniquePtr, IR::MLIRContext::MLIRContext};

/// The MLIR context.
#[derive(Debug)]
pub struct Context {
    pub(crate) inner: RefCell<UniquePtr<MLIRContext>>,
}

impl Context {
    /// Create a new context.
    pub fn new(threaded: bool) -> Self {
        let mut inner = MLIRContext::new();
        inner.pin_mut().enableMultithreading(threaded);

        Self {
            inner: RefCell::new(inner),
        }
    }

    /// Loads all available dialects.
    pub fn load_all_available_dialects(&mut self) {
        self.inner.borrow_mut().pin_mut().loadAllAvailableDialects();
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
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let context = Context::new(true);
        assert!(!context.inner.borrow().is_null());
        assert!(context.is_multithreading_enabled());
    }
}
