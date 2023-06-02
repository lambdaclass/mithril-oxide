# MLIR API design

**Absolute necessities**
  - mlir::IR::Attributes
  - mlir::IR::Block
  - mlir::IR::Location
  - mlir::IR::MLIRContext
  - mlir::IR::Operation
  - mlir::IR::Region
  - mlir::IR::Type
  - mlir::IR::Value

> Question (1): Do we need all its subtypes? (ex. ModuleOp, IntegerType...)

> Question (2): The attributes are its own can of worms. Do we implement the entire tree of types?
    Can we get away with traits and generics (aka. `impl IntoAttribute`)?

> Question (3): Do we make use of lifetimes or just `Rc<_>` the context everywhere?

> Question (4): Some classes can be validated. Should we validate them when building them and return
    a result?


## Question answers

  1. Maybe some of them. For example, we may need to access the associated data with types like
     `memref<?>` or even just plain integers.

  2. I think having both would be nice. Implementing the entire tree would provide introspection
     while having the methods accept `impl IntoAttribute` would make our interface more usable.

  3. Lifetimes are more Rust-like, but may limit the API's usage in some cases. However, I still
     think lifetimes are the way to go.

  4. It'll probably make it easier to spot errors, but we should check its implications (ex. whether
     validating an operation which hasn't been inserted would cause an error).


## Attribute nonsense

  - class Attribute (enum? struct? dyn trait?)
  - The real attributes (structs?, tablegen'd)
  - Attribute interfaces (traits?)

**Proposal**

  - Skip the interface to `mlir::IR::Attribute` (or rather, make it a trait).
  - Make methods that accept attributes have a generic argument which is convertible to the
    attribute.
  - If we make owned values, implement casting methods.
  - Have the common attribute operations (with dyn casting support) in a trait and return a dyn ref
    (or dyn casted concrete structs) on the getters.


## Values

  - Point to either a block argument or an operation result.
  - If bound to the context lifetime only, they'll cause problems when the block or operation is
    removed. They should be bound by the block or value themselves.

**Proposal**

  - There's no need to have `Value` as a concrete type.
  - Have a trait `Value` implemented for block arguments and operation results.
  - Methods that accept arguments will accept `impl Value`.

  - Implication: Having a single-function op builder (like `func::return(...)` in melior) will be
    difficult. Maybe we could use either tuples or a slice of `&dyn Value`.


## Types

  - class Type (enum? struct? dyn trait?)
  - The real types (structs? tablegen'd)
  - Type interfaces (traits?)

**Proposal**

  - Skip the interface to `mlir::IR::Type` (or rather, make it a trait).
  - Make methods that accept attributes have a generic argument which is convertible to the
    attribute.
  - If we make owned values, implement casting methods.
  - Have the common attribute operations (with dyn casting support) in a trait and return a dyn ref
    (or dyn casted concrete structs) on the getters.


## Operations

  - class Operation (enum? struct? dyn trait?)
  - The real operation types ()

**Proposal**

  - Skip the interface to `mlir::IR::Operation` (or rather, make it a trait).
  - Make methods that accept operations have a generic argument which is convertible to the
    operation.
  - If we make owned values, implement casting methods.
  - Have the common attribute operations (with dyn casting support) in a trait and return a dyn ref
    (or dyn casted concrete structs) on the getter.

  - Exception: We need to have the ModuleOp ownable, since it's top-level.


## Example implementation

```rs
pub trait Attribute
where
    Self: Display,
{
    fn downcast<T>(&self) -> Option<&T>;
    fn downcast_mut<T>(&mut self) -> Option<&mut T>;
}
```

## Example usage

```rs
use mithril_oxide::{Context, Module, ops};

let context = Context::new();
let mut module = Module::new(&context, todo!("location"));

let mut region: &mut Region = module.body_mut();

let func_op: &mut ops::func::func = region.op::<ops::func::func>(
    ops::func::func::builder()
        .sym_name("hello")
        .function_type(todo!())
        .sym_visibility(SymVisibility::Public)
        .body(|region| {
            let block: &mut Block = region.emplace_block([]);

            let op0: &ops::arith::constant = body.op(
                ops::arith::constant::builder()
                    .r#type(IntegerType::r#i32())
                    .value(1234)
                    .build()?
            )?;
            body.op(
                ops::func::r#return::builder()
                    .operand(op0.result(0)?)
                    .build()?
            )?;
        })?
        .build()?
);

// func_op.region().blocks().iter();

println!("{module}");
```
