use crate::parsing::{
    CxxBindingKind, CxxForeignAttr, CxxForeignEnum, CxxForeignFn, CxxForeignStruct,
};
use clang::{CallingConvention, Entity, EntityKind, EntityVisitResult};
use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote, TokenStreamExt};
use std::io::{Cursor, Write};
use syn::{FnArg, Pat};

pub fn generate_enum(
    req: &CxxForeignEnum,
    entity: Entity,
) -> Result<(TokenStream, Vec<u8>), Box<dyn std::error::Error>> {
    let mut stream = TokenStream::new();

    let underlying_type = entity
        .get_enum_underlying_type()
        .unwrap()
        .get_canonical_type();
    assert!(underlying_type.is_integer());
    let base_ty = match (
        underlying_type.get_sizeof()?,
        underlying_type.is_signed_integer(),
    ) {
        (1, false) => quote!(u8),
        (1, true) => quote!(i8),
        (2, false) => quote!(u16),
        (2, true) => quote!(i16),
        (4, false) => quote!(u32),
        (4, true) => quote!(i32),
        (8, false) => quote!(u64),
        (8, true) => quote!(i64),
        _ => unreachable!(),
    };

    entity.visit_children(|entity, _| {
        let (value_i64, value_u64) = match entity.get_kind() {
            EntityKind::EnumConstantDecl => entity.get_enum_constant_value(),
            _ => unreachable!(),
        }
        .unwrap();

        let name = format_ident!("{}", entity.get_name().unwrap());
        let value = if underlying_type.is_signed_integer() {
            Literal::i64_unsuffixed(value_i64)
        } else {
            Literal::u64_unsuffixed(value_u64)
        };

        stream.append_all(quote!(#name = #value,));

        EntityVisitResult::Continue
    });

    let attrs = req
        .attrs
        .iter()
        .filter_map(|x| match x {
            CxxForeignAttr::PassThrough(x) => Some(quote!(#x)),
            _ => None,
        })
        .collect::<TokenStream>();
    let vis = &req.vis;
    let ident = &req.ident;

    let stream = quote! {
        #attrs
        #[repr(#base_ty)]
        #vis enum #ident {
            #stream
        }
    };

    Ok((stream, Vec::new()))
}

pub fn generate_fn(
    req: &CxxForeignFn,
    entity: Entity,
) -> Result<(TokenStream, TokenStream, Vec<u8>), Box<dyn std::error::Error>> {
    let mut auxlib = Cursor::new(Vec::new());
    let mangled_name = if entity.is_inline_function() {
        let mangled_name = entity.get_mangled_name().unwrap();

        let has_self = req
            .sig
            .inputs
            .first()
            .is_some_and(|x| matches!(x, FnArg::Receiver(_)));
        let arg_decls = has_self
            .then(|| "self".to_string())
            .into_iter()
            .chain(
                req.sig
                    .inputs
                    .iter()
                    .skip(has_self as _)
                    .zip(entity.get_arguments().unwrap())
                    .map(|(l, r)| match l {
                        FnArg::Typed(x) => match x.pat.as_ref() {
                            Pat::Ident(x) => {
                                format!(
                                    "{} {}",
                                    r.get_type()
                                        .unwrap()
                                        .get_canonical_type()
                                        .get_display_name(),
                                    x.ident
                                )
                            }
                            _ => todo!(),
                        },
                        FnArg::Receiver(_) => unreachable!(),
                    }),
            )
            .intersperse_with(|| ", ".to_string())
            .collect::<String>();
        let arg_names = has_self
            .then(|| "self".to_string())
            .into_iter()
            .chain(
                req.sig
                    .inputs
                    .iter()
                    .skip(has_self as _)
                    .map(|ty| match ty {
                        FnArg::Typed(x) => match x.pat.as_ref() {
                            Pat::Ident(x) => x.ident.to_string(),
                            _ => todo!(),
                        },
                        FnArg::Receiver(_) => unreachable!(),
                    }),
            )
            .intersperse_with(|| ", ".to_string())
            .collect::<String>();

        writeln!(
            auxlib,
            "extern \"C\" {} wrap_{}({}) {{",
            entity.get_result_type().unwrap().get_display_name(),
            mangled_name,
            arg_decls
        )?;
        writeln!(
            auxlib,
            "    return {}({});",
            entity.get_name().unwrap(),
            arg_names
        )?;
        writeln!(auxlib, "}}")?;
        writeln!(auxlib)?;

        format_ident!("wrap_{}", mangled_name)
    } else {
        format_ident!("{}", entity.get_mangled_name().unwrap())
    };

    // Mac OS nonsense.
    #[cfg(target_os = "macos")]
    let mangled_name = mangled_name.strip_prefix('_').unwrap();

    let arg_decls = req
        .sig
        .inputs
        .iter()
        .map(|arg| quote!(#arg,))
        .collect::<TokenStream>();
    let arg_names = req
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Receiver(arg) => match (&arg.mutability, &arg.reference) {
                (None, None) => quote!(&self as *const Self),
                (None, Some(_)) => quote!(self as *const Self),
                (Some(_), None) => quote!(&mut self as *mut Self),
                (Some(_), Some(_)) => quote!(self as *mut Self),
            },
            FnArg::Typed(arg) => match arg.pat.as_ref() {
                Pat::Ident(x) => {
                    let ident = &x.ident;
                    quote!(#ident)
                }
                _ => todo!(),
            },
        })
        .collect::<TokenStream>();
    let ret_ty = &req.sig.output;

    let calling_convention = match entity.get_type().unwrap().get_calling_convention().unwrap() {
        CallingConvention::Cdecl => "C",
        _ => todo!(),
    };
    let vis = &req.vis;
    let ident = &req.sig.ident;

    let decl_stream = quote! {
        extern #calling_convention {
            fn #mangled_name(#arg_decls) #ret_ty;
        }
    };
    let impl_stream = quote! {
        #vis unsafe fn #ident(#arg_decls) #ret_ty {
            #mangled_name(#arg_names)
        }
    };

    Ok((decl_stream, impl_stream, auxlib.into_inner()))
}

pub fn generate_struct(
    req: &CxxForeignStruct,
    entity: Entity,
) -> Result<(TokenStream, Vec<u8>), Box<dyn std::error::Error>> {
    let mut stream = TokenStream::new();

    let cxx_ty = entity.get_type().unwrap();
    let ty_align = Literal::usize_unsuffixed(cxx_ty.get_alignof()?);
    let ty_size = cxx_ty.get_sizeof()?;

    let binding_kind = req
        .attrs
        .iter()
        .find_map(|x| match x {
            CxxForeignAttr::CxxKind(x) => Some(*x),
            _ => None,
        })
        .unwrap_or(CxxBindingKind::OpaqueUnsized);

    match binding_kind {
        CxxBindingKind::OpaqueUnsized => todo!(),
        CxxBindingKind::OpaqueSized => {
            stream.append_all(quote! {
                data: [u8; #ty_size],
                phantom: ::std::marker::PhantomData<::std::marker::PhantomPinned>,
            });
        }
        CxxBindingKind::PartiallyShared => todo!(),
        CxxBindingKind::FullyShared => todo!(),
    };

    let attrs = req
        .attrs
        .iter()
        .filter_map(|x| match x {
            CxxForeignAttr::PassThrough(x) => Some(quote!(#x)),
            _ => None,
        })
        .collect::<TokenStream>();
    let vis = &req.vis;
    let ident = &req.ident;

    let stream = quote! {
        #attrs
        #[repr(C, align(#ty_align))]
        #vis struct #ident {
            #stream
        }
    };

    Ok((stream, Vec::new()))
}
