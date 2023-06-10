#![allow(dead_code)]
#![allow(unused)]

pub use self::{context::Context, location::Location};

pub mod attributes;
mod context;
pub mod location;
pub mod operations;
mod region;
pub mod types;
pub mod util;

// TODO: What are type constraints? Do we need them?
// TODO: What are attribute constraints? Do we need them?

pub mod prelude {
    pub use crate::Context;

    pub use crate::attributes as attrs;
    pub use crate::location as loc;
    pub use crate::operations as ops;
    pub use crate::types;
}
