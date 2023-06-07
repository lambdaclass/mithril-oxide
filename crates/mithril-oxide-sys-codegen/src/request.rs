use syn::{Pat, ReturnType, Type, Visibility};

#[derive(Debug)]
pub struct RequestMod {
    pub includes: Vec<String>,
    pub items: Vec<RequestItem>,
}

#[derive(Debug)]
pub enum RequestItem {
    Struct(RequestStruct),
    Enum(RequestEnum),
    Function(RequestFunction),
}

#[derive(Clone, Debug)]
pub struct RequestStruct {
    pub name: String,
    pub path: String,
    pub kind: RequestStructKind,
    pub vis: Visibility,

    pub items: Vec<RequestMethodImpl>,
}

#[derive(Clone, Debug)]
pub struct RequestEnum {
    pub name: String,
    pub path: String,
    pub vis: Visibility,

    pub variants: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct RequestFunction {
    pub name: String,
    pub cxx_ident: String,
    pub vis: Visibility,
    pub args: Vec<(Option<Pat>, Type)>,
    pub ret: ReturnType,
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub enum RequestMethodImpl {
    Constructor(RequestConstructor),
    Method(RequestMethod),
}

#[derive(Clone, Debug)]
pub struct RequestConstructor {
    pub name: String,
    pub vis: Visibility,
    pub args: Vec<(Pat, Type)>,
}

#[derive(Clone, Debug)]
pub struct RequestMethod {
    pub name: String,
    pub vis: Visibility,
    pub args: Vec<(Option<Pat>, Type)>,
    pub ret: ReturnType,
}
