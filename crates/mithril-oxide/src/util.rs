use crate::Context;

/// Default type and value for builder fields.
pub struct NotSet;

/// Clone of the standard `Into<T>` trait, but with a reference to the MLIR context as an argument.
pub trait FromWithContext<T> {
    fn from_with_context(value: T, context: &Context) -> Self;
}

impl<T> FromWithContext<T> for T {
    fn from_with_context(value: T, _context: &Context) -> Self {
        value
    }
}

#[macro_export]
macro_rules! mlir_asm {
    (
        $block:expr =>
            $( ; $( $( $ret:ident ),+ = )? $op:literal
                ( $( $( $arg:expr ),+ $(,)? )? ) // Values list.
                $( [ $( $( ^ $successor:ident $( ( $( $( $successor_arg:expr ),+ $(,)? )? ) )? ),+ $(,)? )? ] )? // Successors.
                $( < { $( $( $prop_name:pat_param = $prop_value:expr ),+ $(,)? )? } > )? // Properties.
                $( ( $( $( $region:expr ),+ $(,)? )? ) )? // Regions.
                $( { $( $( $attr_name:pat_param = $attr_value:expr ),+ $(,)? )? } )? // Attributes.
                : $args_ty:tt -> $rets_ty:tt // Signature.
                $( loc ( $( $loc:tt )* ) )? // Location.
            )*
    ) => { $(
        $( let $crate::codegen_ret_decl!($($ret),+) = )? {
            let mut builder = $crate::prelude::ops::DynOperation::builder($op);

            let op = $block.push(builder);
            $( $crate::codegen_ret_extr!(op => $($ret),+) )?
        };
    )* };
}

#[doc(hidden)]
#[macro_export]
macro_rules! codegen_ret_decl {
    // Macro entry points.
    ( $ret:ident ) => { $ret };
    ( $( $ret:ident ),+ ) => {
        ( $( codegen_ret_decl!($ret) ),+ )
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! codegen_ret_extr {
    // Macro entry points.
    ( $op:ident => $ret:ident ) => {{
        use $crate::operations::Operation;
        $op.result(0)
    }};
    ( $op:ident => $( $ret:ident ),+ ) => {{
        let mut idx = 0;
        ( $( codegen_ret_extr!(INTERNAL idx, $op => $ret) ),+ )
    }};

    // Internal entry points.
    ( INTERNAL $count:ident, $op:ident => $ret:ident ) => {
        {
            let idx = $count;
            $count += 1;
            $op.result(idx)
        }
    };
}
