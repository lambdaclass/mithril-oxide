pub use self::context::Context;

pub mod attributes;
mod context;
pub mod operations;
pub mod types;

// TODO: What are type constraints? Do we need them?
// TODO: What are attribute constraints? Do we need them?

pub mod prelude {
    pub use crate::Context;

    pub use crate::attributes as attrs;
    pub use crate::operations as ops;
    pub use crate::types;
}
