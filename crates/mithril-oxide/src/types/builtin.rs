use super::{impl_type, impl_type_new, Type};

impl_type!(BFloat16Type);
impl_type!(ComplexType);
impl_type!(Float128Type);
impl_type!(Float16Type);
impl_type!(Float32Type);
impl_type!(Float64Type);
impl_type!(Float80Type);
impl_type!(Float8E4M3B11FNUZType);
impl_type!(Float8E4M3FNType);
impl_type!(Float8E4M3FNUZType);
impl_type!(Float8E5M2FNUZType);
impl_type!(Float8E5M2Type);
impl_type!(FunctionType);
impl_type!(IndexType);
impl_type!(IntegerType);
impl_type!(MemRefType);
impl_type!(NoneType);
impl_type!(OpaqueType);
impl_type!(RankedTensorType);
impl_type!(TupleType);
impl_type!(UnrankedMemRefType);
impl_type!(UnrankedTensorType);
impl_type!(VectorType);

impl_type_new!(BFloat16Type);
impl_type_new!(ComplexType {
    element_type: impl AsRef<Type<'c>>,
});
impl_type_new!(Float128Type);
impl_type_new!(Float16Type);
impl_type_new!(Float32Type);
impl_type_new!(Float64Type);
impl_type_new!(Float80Type);
impl_type_new!(Float8E4M3B11FNUZType);
impl_type_new!(Float8E4M3FNType);
impl_type_new!(Float8E4M3FNUZType);
impl_type_new!(Float8E5M2FNUZType);
impl_type_new!(Float8E5M2Type);
impl_type_new!(FunctionType {
    inputs: impl AsRef<[Type<'c>]>,
    results: impl AsRef<[Type<'c>]>,
});
impl_type_new!(IndexType);
impl_type_new!(IntegerType {
    width: usize,
    signedness: Option<bool>,
});
impl_type_new!(MemRefType {
    shape: impl AsRef<[i64]>,
    element_type: impl AsRef<Type<'c>>,
    // layout: ???,
    // memorySpace: ???,
});
impl_type_new!(NoneType);
// impl_type_new!(OpaqueType);
// impl_type_new!(RankedTensorType);
// impl_type_new!(TupleType);
// impl_type_new!(UnrankedMemRefType);
// impl_type_new!(UnrankedTensorType);
// impl_type_new!(VectorType);
