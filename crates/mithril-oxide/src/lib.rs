#![allow(dead_code)]
#![allow(unused)]
#![feature(concat_idents)]

pub use self::{block::Block, context::Context, location::Location, region::Region};

pub mod attributes;
mod block;
mod context;
pub mod location;
pub mod operations;
mod region;
pub mod types;
pub mod util;
pub mod value;

// TODO: What are type constraints? Do we need them?
// TODO: What are attribute constraints? Do we need them?

pub mod prelude {
    pub use crate::{Block, Context, Region};

    pub use crate::attributes as attrs;
    pub use crate::location as loc;
    pub use crate::operations as ops;
    pub use crate::types;
}
