use crate::{attributes::LocationAttr, operations::OperationBuilder, types::Type, Context};
use std::marker::PhantomData;

pub struct Block<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Block<'c> {
    pub fn add_argument(&mut self, location: impl LocationAttr<'c>, r#type: impl Type<'c>) {
        todo!()
    }

    pub fn push<Op>(&mut self, builder: impl OperationBuilder<'c, Target = Op>) -> &Op {
        // let op = builder.build();
        todo!()
    }
}
