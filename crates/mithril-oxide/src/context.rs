#[derive(Debug)]
pub struct Context {}

impl Context {
    pub fn new(threaded: bool) -> Self {
        todo!()
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new(true)
    }
}
