use quote::ToTokens;
use syn::{
    parse::{discouraged::Speculative, Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
    token, Attribute, Fields, FieldsNamed, FieldsUnnamed, Ident, LitStr, Macro, Meta, Path, Result,
    Signature, Token, Visibility,
};

#[derive(Debug, Clone)]
pub struct CxxForeignMod {
    pub vis: Visibility,
    pub mod_token: Token![mod],
    pub ident: Ident,
    pub brace_token: token::Brace,
    pub items: Vec<CxxForeignItem>,
}

impl Parse for CxxForeignMod {
    fn parse(input: ParseStream) -> Result<Self> {
        let items_buffer;
        Ok(Self {
            vis: input.parse()?,
            mod_token: input.parse()?,
            ident: input.parse()?,
            brace_token: syn::braced!(items_buffer in input),
            items: {
                let mut items = Vec::new();
                while !items_buffer.is_empty() {
                    items.push(items_buffer.parse()?);
                }
                items
            },
        })
    }
}

#[derive(Debug, Clone)]
pub enum CxxForeignItem {
    IncludeAttr(String),

    Enum(CxxForeignEnum),
    Fn(CxxForeignFn),
    Impl(CxxForeignImpl),
    Struct(CxxForeignStruct),
}

impl Parse for CxxForeignItem {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut error;

        let fork_input = input.fork();
        error = match fork_input.parse::<Macro>() {
            Ok(x) => {
                fork_input.parse::<Token![;]>()?;
                input.advance_to(&fork_input);
                if x.path.is_ident("include") {
                    return Ok(Self::IncludeAttr(x.parse_body::<LitStr>()?.value()));
                }
                syn::Error::new(x.path.span(), "Unsupported codegen attribute.")
            }
            Err(e) => e,
        };

        let fork_input = input.fork();
        error.combine(match fork_input.parse::<CxxForeignEnum>() {
            Ok(x) => {
                input.advance_to(&fork_input);
                return Ok(Self::Enum(x));
            }
            Err(e) => e,
        });

        let fork_input = input.fork();
        error.combine(match fork_input.parse::<CxxForeignFn>() {
            Ok(x) => {
                input.advance_to(&fork_input);
                return Ok(Self::Fn(x));
            }
            Err(e) => e,
        });

        let fork_input = input.fork();
        error.combine(match fork_input.parse::<CxxForeignImpl>() {
            Ok(x) => {
                input.advance_to(&fork_input);
                return Ok(Self::Impl(x));
            }
            Err(e) => e,
        });

        let fork_input = input.fork();
        error.combine(match fork_input.parse::<CxxForeignStruct>() {
            Ok(x) => {
                input.advance_to(&fork_input);
                return Ok(Self::Struct(x));
            }
            Err(e) => e,
        });

        Err(error)
    }
}

#[derive(Debug, Clone)]
pub struct CxxForeignEnum {
    pub attrs: Vec<CxxForeignAttr>,
    pub vis: Visibility,
    pub enum_token: Token![enum],
    pub ident: Ident,
    pub brace_token: token::Brace,
    pub variants: Punctuated<Ident, Token![,]>,
}

impl Parse for CxxForeignEnum {
    fn parse(input: ParseStream) -> Result<Self> {
        let variants_buffer;
        Ok(Self {
            attrs: input.call(CxxForeignAttr::parse)?,
            vis: input.parse()?,
            enum_token: input.parse()?,
            ident: input.parse()?,
            brace_token: syn::braced!(variants_buffer in input),
            variants: variants_buffer.parse_terminated(Ident::parse, Token![,])?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CxxForeignFn {
    pub attrs: Vec<CxxForeignAttr>,
    pub vis: Visibility,
    pub sig: Signature,
    pub semi_token: Token![;],
}

impl Parse for CxxForeignFn {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            attrs: input.call(CxxForeignAttr::parse)?,
            vis: input.parse()?,
            sig: input.parse()?,
            semi_token: input.parse()?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CxxForeignImpl {
    pub attrs: Vec<CxxForeignAttr>,
    pub impl_token: Token![impl],
    pub self_ty: Box<Path>,
    pub brace_token: token::Brace,
    pub items: Vec<CxxForeignFn>,
}

impl Parse for CxxForeignImpl {
    fn parse(input: ParseStream) -> Result<Self> {
        let items_buffer;
        Ok(Self {
            attrs: input.call(CxxForeignAttr::parse)?,
            impl_token: input.parse()?,
            self_ty: input.parse()?,
            brace_token: syn::braced!(items_buffer in input),
            items: {
                let mut items = Vec::new();
                while !items_buffer.is_empty() {
                    items.push(items_buffer.parse()?);
                }
                items
            },
        })
    }
}

#[derive(Debug, Clone)]
pub struct CxxForeignStruct {
    pub attrs: Vec<CxxForeignAttr>,
    pub vis: Visibility,
    pub struct_token: Token![struct],
    pub ident: Ident,
    pub fields: Fields,
    pub semi_token: Option<Token![;]>,
}

impl Parse for CxxForeignStruct {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            attrs: input.call(CxxForeignAttr::parse)?,
            vis: input.parse()?,
            struct_token: input.parse()?,
            ident: input.parse()?,
            fields: input.call(|buffer| {
                Ok(if buffer.peek(Token![;]) {
                    Fields::Unit
                } else if buffer.peek(token::Brace) {
                    Fields::Named(buffer.parse::<FieldsNamed>()?)
                } else if buffer.peek(token::Paren) {
                    Fields::Unnamed(buffer.parse::<FieldsUnnamed>()?)
                } else {
                    return Err(syn::Error::new(
                        buffer.span(),
                        "Expected a struct body (or nothing).",
                    ));
                })
            })?,
            semi_token: input.parse()?,
        })
    }
}

#[derive(Debug, Clone)]
pub enum CxxForeignAttr {
    PassThrough(Attribute),
    CxxPath(String),
    CxxKind(CxxBindingKind),
}

impl CxxForeignAttr {
    fn parse(input: ParseStream) -> Result<Vec<Self>> {
        let mut foreign_attrs = Vec::new();
        while input.peek(Token![#]) {
            input
                .call(Attribute::parse_outer)?
                .into_iter()
                .try_for_each::<_, Result<_>>(|attr| {
                    if attr
                        .path()
                        .get_ident()
                        .is_some_and(|x| x.to_string().as_str() == "codegen")
                    {
                        let attr_list = attr
                            .meta
                            .require_list()?
                            .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)?;

                        for inner_attr in attr_list {
                            foreign_attrs.push(if inner_attr.path().is_ident("cxx_path") {
                                Self::CxxPath(
                                    syn::parse2::<LitStr>(
                                        inner_attr.require_name_value()?.value.to_token_stream(),
                                    )?
                                    .value(),
                                )
                            } else if inner_attr.path().is_ident("kind") {
                                Self::CxxKind(syn::parse2(
                                    inner_attr.require_name_value()?.value.to_token_stream(),
                                )?)
                            } else {
                                return Err(syn::Error::new(
                                    inner_attr.span(),
                                    "Expected a valid codegen attribute.",
                                ));
                            });
                        }
                    } else {
                        foreign_attrs.push(CxxForeignAttr::PassThrough(attr));
                    }

                    Ok(())
                })?;
        }

        Ok(foreign_attrs)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CxxBindingKind {
    OpaqueUnsized,
    OpaqueSized,
    PartiallyShared,
    FullyShared,
}

impl Parse for CxxBindingKind {
    fn parse(input: ParseStream) -> Result<Self> {
        let span = input.span();
        Ok(match input.parse::<LitStr>()?.value().as_str() {
            "opaque-unsized" => Self::OpaqueUnsized,
            "opaque-sized" => Self::OpaqueSized,
            "partially-shared" => Self::PartiallyShared,
            "fully-shared" => Self::FullyShared,
            _ => return Err(syn::Error::new(span, "Expected either `opaque-unsized`, `opaque-sized`, `partially-shared` or `fully-shared`.")),
        })
    }
}
