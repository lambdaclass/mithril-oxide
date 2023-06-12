use mithril_oxide_cxx as ffi;

pub struct Region {}

impl Region {
    pub fn new() -> Self {
        ffi::IR::Region
    }
}
