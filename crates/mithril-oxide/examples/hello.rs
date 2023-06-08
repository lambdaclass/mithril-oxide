fn main() {}

// use derive_builder::Builder;
// use mithril_oxide::prelude::*;

// #[derive(Builder, Debug)]
// #[builder(build_fn(skip))]
// struct ConstantOp {
//     #[builder(setter(into))]
//     value: String,
// }

// fn main() {
//     op(ConstantOpBuilder::default().value("3.14159265"));
// }

// trait OpBuilder {
//     type Target;

//     fn build(self) -> Self::Target;
// }

// impl OpBuilder for &mut ConstantOpBuilder {
//     type Target = ConstantOp;

//     fn build(self) -> Self::Target {
//         todo!()
//     }
// }

// fn op<B>(builder: B)
// where
//     B: OpBuilder,
// {
// }
