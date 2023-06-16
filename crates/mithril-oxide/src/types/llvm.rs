use super::{impl_type, impl_type_new, Type};

impl_type!(ArrayType);
impl_type!(FunctionType);
impl_type!(MetadataType);
impl_type!(PpcFp128Type);
impl_type!(PtrType);
impl_type!(StructType);
impl_type!(TokenType);
impl_type!(VectorType);
impl_type!(VoidType);
impl_type!(X86MmxType);

impl_type_new!(X86MmxType);
impl_type_new!(PpcFp128Type);
impl_type_new!(TokenType);
impl_type_new!(MetadataType);
impl_type_new!(VoidType);
impl_type_new!(PtrType {
    pointee: Option<impl AsRef<Type<'c>>>,
    address_space: Option<usize>,
});
impl_type_new!(ArrayType {
    element_type: impl AsRef<Type<'c>>,
    length: usize,
});
impl_type_new!(FunctionType {
    result: impl AsRef<Type<'c>>,
    arguments: impl AsRef<[Type<'c>]>,
    varidic: bool,
});
// impl_type_new!(VectorType {
//     element_type: impl AsRef<Type<'c>>,
//     length: usize,
// });
// impl_type_new!(StructType {
//     fields: impl AsRef<[(&str, Type<'c>)]>,
// });
