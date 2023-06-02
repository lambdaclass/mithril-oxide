use crate::{
    analysis::MappedItemWithWithMethods,
    request::{RequestEnum, RequestMethodImpl, RequestMod, RequestStruct, RequestStructKind},
};
use clang::{CallingConvention, Entity, Type, TypeKind};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, TokenStreamExt};
use std::collections::HashMap;
use syn::LitInt;

pub fn codegen_cpp(request: &RequestMod) -> String {
    let mut output = String::new();

    output.push_str("#include <type_traits>\n\n");
    for include in &request.includes {
        output.push_str(&format!("#include <{include}>\n"));
    }
    output.push_str("\n\n");

    output
}

pub fn codegen_rust(mappings: &HashMap<String, MappedItemWithWithMethods>) -> TokenStream {
    let mut stream = TokenStream::new();

    for (_, mapping) in mappings {
        stream.append_all(match mapping {
            MappedItemWithWithMethods::Struct(request, decl, methods) => {
                codegen_struct(request, decl, methods)
            }
            MappedItemWithWithMethods::Enum(request, decl, variants) => {
                codegen_enum(request, decl, variants)
            }
        });
    }

    stream
}

fn codegen_struct(request: &RequestStruct, decl: &Entity, methods: &[Entity]) -> TokenStream {
    let type_name = format_ident!("{}", request.name);

    let struct_decl = match request.kind {
        RequestStructKind::OpaqueUnsized => todo!(),
        RequestStructKind::OpaqueSized => {
            let ty = decl.get_type().unwrap();
            let vis = &request.vis;

            let align = LitInt::new(&format!("{}", ty.get_alignof().unwrap()), Span::call_site());
            let size = LitInt::new(&format!("{}", ty.get_sizeof().unwrap()), Span::call_site());

            quote! {
                #[repr(C, align(#align))]
                #vis struct #type_name {
                    data: [u8; #size],
                    phantom: ::std::marker::PhantomData<::std::marker::PhantomPinned>,
                }
            }
        }
        RequestStructKind::PartiallyShared => todo!(),
        RequestStructKind::FullyShared => todo!(),
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

                let mangled_name = format_ident!("{}", method.get_mangled_name().unwrap());
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

                let mangled_name = format_ident!("{}", method.get_mangled_name().unwrap());
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

                let calling_convention =
                    match method.get_type().unwrap().get_calling_convention().unwrap() {
                        CallingConvention::Cdecl => "C",
                        _ => panic!(),
                    };

                quote! {
                    extern #calling_convention {
                        fn #mangled_name(this: #self_ty, #args) #ret_ty;
                    }

                    impl #type_name {
                        #vis unsafe fn #name(#self_arg, #args) #ret_ty {
                            #mangled_name(#self_fw, #args_fw)
                        }
                    }
                }
            }
        })
        .collect::<TokenStream>();

    quote! {
        #struct_decl
        #method_impls
    }
}

fn codegen_enum(request: &RequestEnum, decl: &Entity, variants: &[Entity]) -> TokenStream {
    let underlying_type = decl.get_enum_underlying_type().unwrap();

    let name = format_ident!("{}", request.name);
    let vis = &request.vis;
    let ty = codegen_type(&underlying_type);

    let variants = variants
        .iter()
        .map(|x| {
            let name = format_ident!("{}", x.get_display_name().unwrap());
            let value = if underlying_type.is_signed_integer() {
                let x = LitInt::new(
                    &format!("{}", x.get_enum_constant_value().unwrap().0),
                    Span::call_site(),
                );
                quote!(#x)
            } else {
                let x = LitInt::new(
                    &format!("{}", x.get_enum_constant_value().unwrap().1),
                    Span::call_site(),
                );
                quote!(#x)
            };

            quote!(#name = #value,)
        })
        .collect::<TokenStream>();

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
