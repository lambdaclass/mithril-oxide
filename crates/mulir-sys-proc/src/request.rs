use syn::{Type, ReturnType};

#[derive(Debug)]
pub struct RequestMod {
    pub includes: Vec<String>,
    pub items: Vec<RequestItem>,
}

#[derive(Debug)]
pub enum RequestItem {
    Struct(RequestStruct),
}

#[derive(Debug)]
pub struct RequestStruct {
    pub name: String,
    pub path: String,
    pub kind: RequestStructKind,

    pub items: Vec<RequestMethodImpl>,
}

#[derive(Debug)]
pub enum RequestStructKind {
    OpaqueUnsized,
    OpaqueSized,
    PartiallyShared,
    FullyShared,
}

impl TryFrom<&str> for RequestStructKind {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(match value {
            "opaque-unsized" => Self::OpaqueUnsized,
            "opaque-sized" => Self::OpaqueSized,
            "partially-shared" => Self::PartiallyShared,
            "fully-shared" => Self::FullyShared,
            _ => panic!(),
        })
    }
}

#[derive(Debug)]
pub enum RequestMethodImpl {
    Constructor(RequestConstructor),
    Method(RequestMethod),
}

#[derive(Debug)]
pub struct RequestConstructor {
    pub name: String,
    pub args: Vec<Type>,
}

#[derive(Debug)]
pub struct RequestMethod {
    pub name: String,
    pub args: Vec<Type>,
    pub ret: ReturnType,
}
