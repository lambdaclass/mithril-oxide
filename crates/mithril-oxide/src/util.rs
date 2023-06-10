use crate::Context;

/// Default type and value for builder fields.
pub struct NotSet;

/// Clone of the standard `Into<T>` trait, but with a reference to the MLIR context as an argument.
pub trait IntoWithContext<T> {
    fn into_with_context(self, context: &Context) -> T;
}

impl<T> IntoWithContext<T> for T {
    fn into_with_context(self, _context: &Context) -> T {
        self
    }
}

macro_rules! mlir_asm {
    (
        block =>
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
        $( let codegen_ret_decl!($($ret),+) = )? {
            let op = todo!();

            $( codegen_ret_extr!(op => $($ret),+) )?
        };
    )* };
}

macro_rules! codegen_ret_decl {
    // Macro entry points.
    ( $ret:ident ) => { $ret };
    ( $( $ret:ident ),+ ) => {
        ( $( codegen_ret_decl!($ret) ),+ )
    };
}

macro_rules! codegen_ret_extr {
    // Macro entry points.
    ( $op:ident => $ret:ident ) => { $op.result($ret) };
    ( $op:ident => $( $ret:ident ),+ ) => {
        ( $( codegen_ret_extr!($op => $ret) ),+ )
    };
}

// #[cfg(test)]
// mod test {
//     fn test() {
//         mlir_asm! { block =>
//             ; t0 = "arith.constant"() { value = 0 } : () -> i252
//             ; t0, t1 = "arith.constant"() { value = 0 } : () -> i252
//             ; "func.return"(t0) : (i252) -> ()
//         }
//     }
// }
