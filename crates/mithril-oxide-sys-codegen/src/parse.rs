//! Rust macro input parsing.

use crate::request::{
    RequestConstructor, RequestEnum, RequestFunction, RequestItem, RequestMethod,
    RequestMethodImpl, RequestMod, RequestStruct, RequestStructKind,
};
use syn::{
    punctuated::Punctuated, Fields, ForeignItemFn, ItemEnum, ItemFn, ItemImpl, ItemMod, ItemStruct,
    MetaNameValue, Token, Type,
};

pub fn parse_macro_input(input: ItemMod) -> Result<RequestMod, Box<dyn std::error::Error>> {
    let mut target = RequestMod {
        includes: Vec::new(),
        items: Vec::new(),
    };

    for attr in input.attrs {
        if attr.path().is_ident("codegen") {
            let items: Punctuated<MetaNameValue, Token![,]> = attr
                .parse_args_with(Punctuated::parse_separated_nonempty)
                .unwrap();
            for item in items {
                match () {
                    _ if item.path.is_ident("include") => target.includes.push(match item.value {
                        syn::Expr::Lit(x) => match x.lit {
                            syn::Lit::Str(x) => x.value(),
                            _ => panic!(),
                        },
                        _ => panic!(),
                    }),
                    _ => panic!(),
                }
            }
        } else {
            panic!()
        }
    }

    let (_, content) = input.content.unwrap();
    for item in content {
        match item {
            syn::Item::Const(_) => todo!(),
            syn::Item::Enum(x) => parse_enum(&mut target, x)?,
            syn::Item::ExternCrate(_) => todo!(),
            syn::Item::Fn(x) => parse_fn(&mut target, x)?,
            syn::Item::ForeignMod(_) => todo!(),
            syn::Item::Impl(x) => parse_impl(&mut target, x)?,
            syn::Item::Macro(_) => todo!(),
            syn::Item::Mod(_) => todo!(),
            syn::Item::Static(_) => todo!(),
            syn::Item::Struct(x) => parse_struct(&mut target, x)?,
            syn::Item::Trait(_) => todo!(),
            syn::Item::TraitAlias(_) => todo!(),
            syn::Item::Type(_) => todo!(),
            syn::Item::Union(_) => todo!(),
            syn::Item::Use(_) => todo!(),
            syn::Item::Verbatim(_) => todo!(),
            _ => todo!(),
        }
    }

    Ok(target)
}

fn parse_struct(
    parent: &mut RequestMod,
    item: ItemStruct,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut target = RequestStruct {
        name: item.ident.to_string(),
        path: item.ident.to_string(),
        kind: RequestStructKind::OpaqueUnsized,
        vis: item.vis,
        items: Vec::new(),
    };

    for attr in &item.attrs {
        if attr.path().is_ident("codegen") {
            let items: Punctuated<MetaNameValue, Token![,]> = attr
                .parse_args_with(Punctuated::parse_separated_nonempty)
                .unwrap();
            for item in items {
                match () {
                    _ if item.path.is_ident("cxx_path") => {
                        target.path = match item.value {
                            syn::Expr::Lit(x) => match x.lit {
                                syn::Lit::Str(x) => x.value(),
                                _ => panic!(),
                            },
                            _ => panic!(),
                        }
                    }
                    _ if item.path.is_ident("kind") => {
                        target.kind = match item.value {
                            syn::Expr::Lit(x) => match x.lit {
                                syn::Lit::Str(x) => x.value().as_str().try_into().unwrap(),
                                _ => panic!(),
                            },
                            _ => panic!(),
                        }
                    }
                    _ => panic!("{:?}", item.path),
                }
            }
        } else {
            // TODO: Support forwarding other attributes.
            panic!();
        }
    }

    match target.kind {
        RequestStructKind::OpaqueUnsized | RequestStructKind::OpaqueSized => {
            assert_eq!(item.fields, Fields::Unit);
        }
        RequestStructKind::PartiallyShared => {
            assert!(matches!(item.fields, Fields::Named(_)));
            todo!("support fields")
        }
        RequestStructKind::FullyShared => {
            assert!(!matches!(item.fields, Fields::Unnamed(_)));
            todo!("support fields")
        }
    }

    parent.items.push(RequestItem::Struct(target));
    Ok(())
}

fn parse_enum(parent: &mut RequestMod, item: ItemEnum) -> Result<(), Box<dyn std::error::Error>> {
    let mut target = RequestEnum {
        name: item.ident.to_string(),
        path: item.ident.to_string(),
        vis: item.vis,
        variants: Vec::new(),
    };

    for attr in &item.attrs {
        if attr.path().is_ident("codegen") {
            let items: Punctuated<MetaNameValue, Token![,]> = attr
                .parse_args_with(Punctuated::parse_separated_nonempty)
                .unwrap();
            for item in items {
                match () {
                    _ if item.path.is_ident("cxx_path") => {
                        target.path = match item.value {
                            syn::Expr::Lit(x) => match x.lit {
                                syn::Lit::Str(x) => x.value(),
                                _ => panic!(),
                            },
                            _ => panic!(),
                        };
                    }
                    _ => panic!("{:?}", item.path),
                }
            }
        } else {
            // TODO: Support forwarding other attributes.
            panic!();
        }
    }

    for variant in item.variants {
        assert_eq!(variant.fields, Fields::Unit);
        target.variants.push(variant.ident.to_string());
    }

    parent.items.push(RequestItem::Enum(target));
    Ok(())
}

fn parse_impl(
    parent: &mut RequestMod,
    root_item: ItemImpl,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut target = Vec::new();

    let self_ty = match Box::as_ref(&root_item.self_ty) {
        Type::Array(_) => todo!(),
        Type::BareFn(_) => todo!(),
        Type::Group(_) => todo!(),
        Type::ImplTrait(_) => todo!(),
        Type::Infer(_) => todo!(),
        Type::Macro(_) => todo!(),
        Type::Never(_) => todo!(),
        Type::Paren(_) => todo!(),
        Type::Path(x) => x.path.get_ident().unwrap().to_string(),
        Type::Ptr(_) => todo!(),
        Type::Reference(_) => todo!(),
        Type::Slice(_) => todo!(),
        Type::TraitObject(_) => todo!(),
        Type::Tuple(_) => todo!(),
        Type::Verbatim(_) => todo!(),
        _ => todo!(),
    };

    for item in root_item.items {
        let item = match item {
            syn::ImplItem::Verbatim(x) => syn::parse2::<ForeignItemFn>(x).unwrap(),
            _ => todo!(),
        };

        let mut is_constructor = false;
        for attr in item.attrs {
            if attr.path().is_ident("codegen") {
                let item = attr.parse_args::<syn::Ident>().unwrap();
                if item == "constructor" {
                    is_constructor = true;
                } else {
                    panic!()
                }
            }
        }

        if is_constructor {
            assert!(match item.sig.output {
                syn::ReturnType::Type(_, ty) => {
                    ty == root_item.self_ty || *ty == syn::parse_quote!(Self)
                }
                syn::ReturnType::Default => todo!(),
            });
            target.push(RequestMethodImpl::Constructor(RequestConstructor {
                name: item.sig.ident.to_string(),
                vis: item.vis,
                args: item
                    .sig
                    .inputs
                    .into_iter()
                    .map(|x| match x {
                        syn::FnArg::Receiver(_) => panic!(),
                        syn::FnArg::Typed(x) => (*x.pat, *x.ty),
                    })
                    .collect(),
            }));
        } else {
            target.push(RequestMethodImpl::Method(RequestMethod {
                name: item.sig.ident.to_string(),
                vis: item.vis,
                args: item
                    .sig
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(i, x)| match x {
                        syn::FnArg::Receiver(_x) => {
                            assert_eq!(i, 0);
                            // assert_eq!(*x.ty, syn::parse_quote!(&mut self));
                            (None, (*root_item.self_ty).clone())
                        }
                        syn::FnArg::Typed(x) => (Some((*x.pat).clone()), (*x.ty).clone()),
                    })
                    .collect(),
                ret: item.sig.output,
            }));
        }
    }

    let parent = parent
        .items
        .iter_mut()
        .find_map(|x| match x {
            RequestItem::Struct(x) => (x.name == self_ty).then_some(x),
            _ => None,
        })
        .unwrap();
    parent.items.extend(target.into_iter());
    Ok(())
}

fn parse_fn(parent: &mut RequestMod, item: ItemFn) -> Result<(), Box<dyn std::error::Error>> {
    let mut target = RequestFunction {
        name: item.sig.ident.to_string(),
        vis: item.vis,
        cxx_ident: item.sig.ident.to_string(),
        args: item
            .sig
            .inputs
            .iter()
            .enumerate()
            .map(|(i, x)| match x {
                syn::FnArg::Receiver(_x) => {
                    todo!()
                }
                syn::FnArg::Typed(x) => (Some((*x.pat).clone()), (*x.ty).clone()),
            })
            .collect(),
        ret: item.sig.output,
    };

    for attr in &item.attrs {
        if attr.path().is_ident("codegen") {
            let items: Punctuated<MetaNameValue, Token![,]> = attr
                .parse_args_with(Punctuated::parse_separated_nonempty)
                .unwrap();
            for item in items {
                match () {
                    _ if item.path.is_ident("cxx_ident") => {
                        target.cxx_ident = match item.value {
                            syn::Expr::Lit(x) => match x.lit {
                                syn::Lit::Str(x) => x.value(),
                                _ => panic!(),
                            },
                            _ => panic!(),
                        };
                    }
                    _ => panic!("{:?}", item.path),
                }
            }
        } else {
            // TODO: Support forwarding other attributes.
            panic!();
        }
    }

    parent.items.push(RequestItem::Function(target));
    Ok(())
}
