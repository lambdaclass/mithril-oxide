use crate::parsing::{
    CxxBindingKind, CxxForeignAttr, CxxForeignEnum, CxxForeignFn, CxxForeignStruct,
};
use clang::{Entity, EntityKind, EntityVisitResult};
use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote, TokenStreamExt};
use std::io::Cursor;

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
) -> Result<(TokenStream, Vec<u8>), Box<dyn std::error::Error>> {
    let mut stream = TokenStream::new();
    let mut auxlib = Cursor::new(Vec::new());

    let mangled_name = if entity.is_inline_function() {
        // todo!("Write auxiliary wrapper.")
        entity.get_mangled_name().unwrap()
    } else {
        entity.get_mangled_name().unwrap()
    };

    Ok((stream, auxlib.into_inner()))
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
