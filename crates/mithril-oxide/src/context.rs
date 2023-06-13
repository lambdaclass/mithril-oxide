use mithril_oxide_sys::{UniquePtr, IR::MLIRContext::MLIRContext};

#[derive(Debug)]
pub struct Context {
    inner: UniquePtr<MLIRContext>,
}

impl Context {
    pub fn new(threaded: bool) -> Self {
        todo!()
    }
}

impl Default for Context {
    fn default() -> Self {
        Self {
            inner: MLIRContext::new(),
        }
    }
}
