use super::{impl_type, impl_type_new, Type};

impl_type!(CoroHandleType);
impl_type!(CoroIdType);
impl_type!(CoroStateType);
impl_type!(GroupType);
impl_type!(TokenType);
impl_type!(ValueType);

impl_type_new!(CoroHandleType);
impl_type_new!(CoroIdType);
impl_type_new!(CoroStateType);
impl_type_new!(GroupType);
impl_type_new!(TokenType);
impl_type_new!(ValueType {
    value_type: impl AsRef<Type<'c>>,
});
