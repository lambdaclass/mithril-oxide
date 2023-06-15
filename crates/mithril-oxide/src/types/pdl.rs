use super::{impl_type, impl_type_new, Type};

impl_type!(AttributeType);
impl_type!(OperationType);
impl_type!(RangeType);
impl_type!(TypeType);
impl_type!(ValueType);

impl_type_new!(AttributeType);
impl_type_new!(OperationType);
impl_type_new!(RangeType {
    element_type: impl AsRef<Type<'c>>,
});
impl_type_new!(TypeType);
impl_type_new!(ValueType);
