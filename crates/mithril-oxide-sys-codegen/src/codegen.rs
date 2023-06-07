use crate::parsing::{CxxForeignAttr, CxxForeignEnum, CxxForeignFn, CxxForeignStruct};
use clang::{Entity, EntityKind, EntityVisitResult};
use proc_macro2::TokenStream;
use quote::{format_ident, quote, TokenStreamExt};
use std::io::Cursor;
use syn::FnArg;

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

    let method_impls = request
        .items
        .iter()
        .zip(methods)
        .map(|(method_decl, method)| match method_decl {
            RequestMethodImpl::Constructor(request) => {
                let name = format_ident!("{}", request.name);
                let vis = &request.vis;
                let args = request
                    .args
                    .iter()
                    .map(|(pat, ty)| quote!(#pat: #ty,))
                    .collect::<TokenStream>();

                let mangled_name = match method.get_mangled_name().unwrap() {
                    #[cfg(target_os = "macos")]
                    x if x.starts_with('_') => {
                        format_ident!("{}", x.strip_prefix('_').unwrap())
                    }
                    x => format_ident!("{}", x),
                };
                let args_fw = request
                    .args
                    .iter()
                    .map(|(pat, _)| quote!(#pat,))
                    .collect::<TokenStream>();

                let calling_convention =
                    match method.get_type().unwrap().get_calling_convention().unwrap() {
                        CallingConvention::Cdecl => "C",
                        _ => panic!(),
                    };

                // TODO: Handle different binding kinds.
                quote! {
                    extern #calling_convention {
                        fn #mangled_name(this: *mut #type_name, #args);
                    }

                    impl #type_name {
                        #vis unsafe fn #name(#args) -> Self {
                            let mut this = std::mem::MaybeUninit::<#type_name>::uninit();
                            #mangled_name(this.as_mut_ptr(), #args_fw);
                            this.assume_init()
                        }
                    }
                }
            }
            RequestMethodImpl::Method(request) => {
                let name = format_ident!("{}", request.name);
                let vis = &request.vis;
                let args = request
                    .args
                    .iter()
                    .skip(1)
                    .map(|(pat, ty)| quote!(#pat: #ty,))
                    .collect::<TokenStream>();

                let mangled_name = match method.get_mangled_name().unwrap() {
                    #[cfg(target_os = "macos")]
                    x if x.starts_with('_') => {
                        format_ident!("{}", x.strip_prefix('_').unwrap())
                    }
                    x => format_ident!("{}", x),
                };
                let args_fw = request
                    .args
                    .iter()
                    .skip(1)
                    .map(|(pat, _)| quote!(#pat,))
                    .collect::<TokenStream>();

                let ret_ty = &request.ret;
                let (self_arg, self_fw, self_ty) = if method.is_const_method() {
                    (
                        quote!(&self),
                        quote!(self as *const _),
                        quote!(*const #type_name),
                    )
                } else {
                    (
                        quote!(&mut self),
                        quote!(self as *mut _),
                        quote!(*mut #type_name),
                    )
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

    quote! {
        #[repr(#ty)]
        #vis enum #name {
            #variants
        }
    }
}

fn codegen_type(ty: &Type) -> TokenStream {
    match ty.get_kind() {
        TypeKind::Int => quote!(i32),
        _ => todo!(),
    }
}

fn codegen_func(request: &RequestFunction, decl: &Entity) -> TokenStream {
    let name = format_ident!("{}", request.name);
    let vis = &request.vis;
    let args = request
        .args
        .iter()
        .map(|(pat, ty)| quote!(#pat: #ty,))
        .collect::<TokenStream>();

    let mangled_name = match decl.get_mangled_name().unwrap() {
        #[cfg(target_os = "macos")]
        x if x.starts_with('_') => {
            format_ident!("{}", x.strip_prefix('_').unwrap())
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
        todo!("Write auxiliary wrapper.")
    } else {
        entity.get_mangled_name().unwrap()
    };

    assert_eq!(
        req.sig.inputs.len(),
        entity.get_arguments().unwrap().len(),
        "Wrong number of arguments."
    );
    if req
        .sig
        .inputs
        .first()
        .is_some_and(|x| matches!(x, FnArg::Receiver(_)))
    {
        assert_eq!(entity.get_kind(), EntityKind::Method);
        assert!(!entity.is_static_method());
    }

    Ok((stream, auxlib.into_inner()))
}

pub fn generate_struct(
    req: &CxxForeignStruct,
    entity: Entity,
) -> Result<(TokenStream, Vec<u8>), Box<dyn std::error::Error>> {
    todo!()
}
