use crate::{attributes::LocationAttr, types::Type, Context};
use std::marker::PhantomData;

pub struct Block<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Block<'c> {
    pub fn new() -> Self {
        todo!()
    }

    pub fn add_argument(&mut self, location: impl LocationAttr<'c>, r#type: impl Type<'c>) {
        todo!()
    }
}
