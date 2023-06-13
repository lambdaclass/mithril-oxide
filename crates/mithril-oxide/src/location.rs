use std::{cell::RefCell, fmt::Display, marker::PhantomData};

use mithril_oxide_sys::{UniquePtr, IR};

use crate::Context;

pub struct Location<'ctx> {
    inner: RefCell<UniquePtr<IR::Location::Location>>,
    _ctx: PhantomData<&'ctx Context>,
}

impl<'ctx> Location<'ctx> {
    pub fn unknown(ctx: &'ctx mut Context) -> Location<'ctx> {
        let loc = IR::Location::UnknownLoc::get(ctx.inner.borrow_mut().pin_mut());
        Self {
            inner: RefCell::new(loc.as_ref().unwrap().into()),
            _ctx: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown() {
        let mut context = Context::new(true);
        let loc = Location::unknown(&mut context);
        assert!(!loc.inner.borrow().is_null());
    }
}
