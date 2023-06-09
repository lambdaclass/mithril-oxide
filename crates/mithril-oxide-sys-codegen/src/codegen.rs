use crate::parsing::{
    CxxBindingKind, CxxForeignAttr, CxxForeignEnum, CxxForeignFn, CxxForeignStruct,
};
use clang::{CallingConvention, Entity, EntityKind, EntityVisitResult};
use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote, TokenStreamExt};
use std::{
    borrow::Cow,
    io::{Cursor, Write},
};
use syn::{FnArg, Ident, Pat, ReturnType};

#[allow(clippy::similar_names)]
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

#[allow(clippy::too_many_lines)]
pub fn generate_fn(
    req: &CxxForeignFn,
    entity: Entity,
    self_ty: Option<(&Ident, &str)>,
    _auxlib_name: &str,
) -> Result<(TokenStream, TokenStream, Vec<u8>), Box<dyn std::error::Error>> {
    let mut auxlib = Cursor::new(Vec::new());
    let (mangled_name, link_attr) = if entity.is_inline_function() {
        let mangled_name = entity.get_mangled_name().unwrap();

        let has_self = req
            .sig
            .inputs
            .first()
            .is_some_and(|x| matches!(x, FnArg::Receiver(_)));
        let arg_decls = has_self
            .then(|| format!("{} self", self_ty.unwrap().1))
            .into_iter()
            .chain(
                req.sig
                    .inputs
                    .iter()
                    .skip(usize::from(has_self))
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
        let args_without_self = req
            .sig
            .inputs
            .iter()
            .skip(usize::from(has_self))
            .map(|ty| match ty {
                FnArg::Typed(x) => match x.pat.as_ref() {
                    Pat::Ident(x) => x.ident.to_string(),
                    _ => todo!(),
                },
                FnArg::Receiver(_) => unreachable!(),
            })
            .collect::<Vec<_>>();
        let arg_names = has_self
            .then_some("self")
            .into_iter()
            .chain(args_without_self.iter().map(String::as_str))
            .intersperse(", ")
            .collect::<String>();

        writeln!(
            auxlib,
            "extern \"C\" {} wrap_{}({}) {{",
            entity
                .get_result_type()
                .unwrap()
                .get_canonical_type()
                .get_display_name(),
            mangled_name,
            arg_decls
        )?;
        match entity.get_kind() {
            EntityKind::Constructor => {
                writeln!(
                    auxlib,
                    "    new({}) {}({});",
                    if req.sig.inputs.len() == 1 {
                        arg_names.as_str()
                    } else {
                        "self"
                    },
                    self_ty.unwrap().1,
                    args_without_self
                        .iter()
                        .map(String::as_str)
                        .intersperse(", ")
                        .collect::<String>()
                )?;
                writeln!(auxlib, "    return;")?;
            }
            EntityKind::Destructor | EntityKind::Method => {
                if entity.is_static_method() {
                    writeln!(
                        auxlib,
                        "    return {}::{}({});",
                        self_ty.unwrap().1,
                        entity.get_name().unwrap(),
                        arg_names
                    )?;
                } else {
                    writeln!(
                        auxlib,
                        "    return self.{}({});",
                        entity.get_name().unwrap(),
                        args_without_self
                            .iter()
                            .map(String::as_str)
                            .intersperse(", ")
                            .collect::<String>()
                    )?;
                }
            }
            EntityKind::FunctionDecl => writeln!(
                auxlib,
                "    return {}({});",
                entity.get_name().unwrap(),
                arg_names
            )?,
            _ => unreachable!(),
        }
        writeln!(auxlib, "}}")?;
        writeln!(auxlib)?;

        // // Mac OS nonsense.
        // #[cfg(not(target_os = "macos"))]
        // let cpp_runtime = "stdc++";
        // #[cfg(target_os = "macos")]
        // let cpp_runtime = "c++";

        (
            format_ident!("wrap_{}", mangled_name),
            quote! {
                // #[link(name = #auxlib_name, kind = "static")]
                // #[link(name = #cpp_runtime)]
            },
        )
    } else {
        (
            format_ident!("{}", entity.get_mangled_name().unwrap()),
            TokenStream::new(),
        )
    };

    // Mac OS nonsense.
    #[cfg(target_os = "macos")]
    let mangled_name = match mangled_name.to_string() {
        x if x.starts_with("wrap_") => mangled_name,
        x => format_ident!("{}", x.strip_prefix('_').unwrap().to_string()),
    };

    let extern_arg_decls = req
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Receiver(x) => {
                let self_ty = self_ty.unwrap().0;
                match &x.mutability {
                    None => quote!(this: *const #self_ty,),
                    Some(_) => quote!(this: *mut #self_ty,),
                }
            }
            arg @ FnArg::Typed(_) => quote!(#arg,),
        })
        .collect::<TokenStream>();
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
                (None, None) => quote!(&self as *const Self,),
                (None, Some(_)) => quote!(self as *const Self,),
                (Some(_), None) => quote!(&mut self as *mut Self,),
                (Some(_), Some(_)) => quote!(self as *mut Self,),
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
    let ret_ty = if req.sig.output == syn::parse_quote!(-> Self) {
        let x = self_ty.unwrap().0;
        Cow::Owned(syn::parse_quote!(-> #x))
    } else {
        Cow::Borrowed(&req.sig.output)
    };

    let calling_convention = match entity.get_type().unwrap().get_calling_convention().unwrap() {
        CallingConvention::Cdecl => "C",
        _ => todo!(),
    };
    let vis = &req.vis;
    let ident = &req.sig.ident;

    let (decl_stream, impl_stream) = match entity.get_kind() {
        EntityKind::Constructor => {
            let self_ty = self_ty.unwrap().0;
            (
                quote! {
                    #link_attr
                    extern #calling_convention {
                        fn #mangled_name(this: *mut #self_ty, #extern_arg_decls);
                    }
                },
                quote! {
                    #vis unsafe fn #ident(#arg_decls) #ret_ty {
                        let mut this = ::std::mem::MaybeUninit::<#self_ty>::uninit();
                        #mangled_name(this.as_mut_ptr(), #arg_names);
                        this.assume_init()
                    }
                },
            )
        }
        EntityKind::Destructor => {
            assert_eq!(req.sig.inputs.len(), 1);
            assert!(matches!(ret_ty.as_ref(), &ReturnType::Default));

            (
                quote! {
                    #link_attr
                    extern #calling_convention {
                        fn #mangled_name(#extern_arg_decls);
                    }
                },
                quote! {
                    #vis unsafe fn #ident(#arg_decls) {
                        #mangled_name(&mut self as *mut _);
                    }
                },
            )
        }
        _ => (
            quote! {
                #link_attr
                extern #calling_convention {
                    fn #mangled_name(#extern_arg_decls) #ret_ty;
                }
            },
            quote! {
                #vis unsafe fn #ident(#arg_decls) #ret_ty {
                    #mangled_name(#arg_names)
                }
            },
        ),
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
