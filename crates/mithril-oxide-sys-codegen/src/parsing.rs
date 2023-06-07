use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token, Attribute, Fields, FieldsNamed, FieldsUnnamed, Ident, LitStr, Macro, Path, Result,
    Signature, Token, Visibility,
};

#[derive(Debug, Clone)]
pub struct CxxForeignMod {
    pub abi: Token![extern],
    pub brace_token: token::Brace,
    pub items: Vec<CxxForeignItem>,
}

impl Parse for CxxForeignMod {
    fn parse(input: ParseStream) -> Result<Self> {
        let attr = input.call(Attribute::parse_outer)?;
        assert_eq!(attr.len(), 1);
        assert_eq!(
            attr[0].meta.require_path_only()?.get_ident().unwrap(),
            "codegen"
        );

        let items_buffer;
        Ok(Self {
            abi: input.parse()?,
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

        error = match input.parse::<Macro>() {
            Ok(x) => {
                input.parse::<Token![;]>()?;
                if x.path.is_ident("include") {
                    return Ok(Self::IncludeAttr(x.parse_body::<LitStr>()?.value()));
                } else {
                    panic!("todo: throw error")
                }
            }
            Err(e) => e,
        };

        error.combine(match input.parse::<CxxForeignEnum>() {
            Ok(x) => return Ok(Self::Enum(x)),
            Err(e) => e,
        });

        error.combine(match input.parse::<CxxForeignFn>() {
            Ok(x) => return Ok(Self::Fn(x)),
            Err(e) => e,
        });

        error.combine(match input.parse::<CxxForeignImpl>() {
            Ok(x) => return Ok(Self::Impl(x)),
            Err(e) => e,
        });

        error.combine(match input.parse::<CxxForeignStruct>() {
            Ok(x) => return Ok(Self::Struct(x)),
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
            items: items_buffer
                .call(Punctuated::<_, Token![;]>::parse_terminated)?
                .iter()
                .cloned()
                .collect(),
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
                } else if buffer.peek(token::Brace) {
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
                        let attr = attr.meta.require_list()?;
                        attr.parse_nested_meta(|meta| {
                            foreign_attrs.push(
                                match meta.path.get_ident().map(|x| x.to_string()).as_deref() {
                                    Some("cxx_path") => {
                                        Self::CxxPath(meta.value()?.parse::<LitStr>()?.value())
                                    }
                                    Some("kind") => Self::CxxKind(meta.value()?.parse()?),
                                    _ => {
                                        return Err(syn::Error::new(
                                            meta.input.span(),
                                            "Expected a valid codegen attribute.",
                                        ))
                                    }
                                },
                            );
                            Ok(())
                        })?;
                    } else {
                        foreign_attrs.push(CxxForeignAttr::PassThrough(attr))
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

fn parse_include_attr(input: ParseStream) -> Result<Vec<String>> {
    input
        .call(Attribute::parse_inner)?
        .into_iter()
        .map(|attr| {
            let meta = attr.meta.require_list()?;
            Ok(if meta.path.is_ident("codegen") {
                let mut value = None;
                meta.parse_nested_meta(|nested_meta| {
                    if nested_meta.path.is_ident("include") {
                        value = Some(nested_meta.value()?.parse::<LitStr>()?.value());
                    } else {
                        panic!("todo: throw error");
                    }

                    Ok(())
                })?;

                value.unwrap()
            } else {
                panic!("todo: throw error")
            })
        })
        .try_collect()
}
