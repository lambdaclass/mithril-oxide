use super::{impl_type, impl_type_new, Type};

impl_type!(AnyOpType);
impl_type!(OperationType);
impl_type!(ParamType);

impl_type_new!(AnyOpType);
impl_type_new!(OperationType {
    operation_name: impl AsRef<str>,
});
impl_type_new!(ParamType {
    r#type: impl AsRef<Type<'c>>,
});
