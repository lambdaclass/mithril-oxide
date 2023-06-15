use super::{impl_type, impl_type_new, Type};

impl_type!(OpaqueType);
impl_type!(PointerType);

impl_type_new!(OpaqueType {
    value: impl AsRef<str>,
});
impl_type_new!(PointerType {
    pointee: impl AsRef<Type<'c>>,
});
