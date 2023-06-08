#![deny(clippy::pedantic)]
#![deny(warnings)]
#![feature(iter_intersperse)]
#![feature(iterator_try_collect)]

use crate::parsing::{CxxForeignAttr, CxxForeignItem, CxxForeignMod};
use clang::{Clang, EntityKind, Index};
use proc_macro2::TokenStream;
use quote::{quote, TokenStreamExt};
use std::{
    borrow::Cow,
    collections::HashMap,
    fs::File,
    io::{self, Write},
    path::Path,
};
use tempfile::tempdir;

mod analysis;
mod codegen;
mod parsing;
mod wrappers;

#[allow(clippy::missing_errors_doc)]
#[allow(clippy::missing_panics_doc)]
#[allow(clippy::too_many_lines)]
pub fn codegen(
    auxlib_path: &Path,
    stream: TokenStream,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    let foreign_mod: CxxForeignMod = syn::parse2(stream)?;

    let clang = Clang::new()?;
    let index = Index::new(&clang, true, false);

    let temp_dir = tempdir()?;
    let ast_source_path = temp_dir.path().join("ast.cpp");
    let aux_source_path = temp_dir.path().join("aux.cpp");

    // Write C++ source for analysis.
    foreign_mod
        .items
        .iter()
        .filter_map(|x| match x {
            CxxForeignItem::IncludeAttr(x) => Some(x),
            _ => None,
        })
        .try_fold::<_, _, io::Result<_>>(File::create(&ast_source_path)?, |mut f, i| {
            writeln!(f, "#include <{i}>")?;
            Ok(f)
        })?;

    let translation_unit = analysis::parse_cpp(&index, &ast_source_path)?;

    let mut out_stream = TokenStream::new();
    let mut aux_source = File::create(&aux_source_path)?;
    let mut aux_source_required = false;

    // Insert include statements into the aux library.
    for item in &foreign_mod.items {
        if let CxxForeignItem::IncludeAttr(file) = item {
            writeln!(aux_source, "#include <{file}>")?;
        }
    }
    writeln!(aux_source)?;

    // Process types and generate their mappings.
    let mut mappings = HashMap::new();
    for item in &foreign_mod.items {
        match item {
            CxxForeignItem::Enum(req) => {
                let cxx_path = find_cxx_path(&req.attrs)
                    .map_or_else(|| Cow::Owned(req.ident.to_string()), Cow::Borrowed);
                let entity =
                    analysis::find_enum(&translation_unit, &cxx_path).expect("Entity not found");

                mappings.insert(req.ident.clone(), cxx_path.to_string());

                let (out_chunk, aux_chunk) = codegen::generate_enum(req, entity)?;
                out_stream.append_all(out_chunk);
                aux_source.write_all(&aux_chunk)?;
            }
            CxxForeignItem::Struct(req) => {
                let cxx_path = find_cxx_path(&req.attrs)
                    .map_or_else(|| Cow::Owned(req.ident.to_string()), Cow::Borrowed);
                let entity =
                    analysis::find_struct(&translation_unit, &cxx_path).expect("Entity not found");

                mappings.insert(req.ident.clone(), cxx_path.to_string());

                let (out_chunk, aux_chunk) = codegen::generate_struct(req, entity)?;
                out_stream.append_all(out_chunk);
                aux_source.write_all(&aux_chunk)?;
            }
            _ => {}
        }
    }

    // Process functions and methods using the precomputed mappings.
    let auxlib_name = auxlib_path
        .with_extension("")
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
        .strip_prefix("lib")
        .unwrap()
        .to_string();

    let mut ffi_stream = out_stream;
    let mut out_stream = TokenStream::new();
    for item in &foreign_mod.items {
        match item {
            CxxForeignItem::Fn(req) => {
                let entity = analysis::find_fn(
                    &translation_unit,
                    find_cxx_path(&req.attrs).unwrap_or(&req.sig.ident.to_string()),
                    &req.sig,
                    &mappings,
                )
                .expect("Entity not found");
                assert_eq!(
                    entity.get_kind(),
                    EntityKind::FunctionDecl,
                    "Non-impl methods (functions with self) are not allowed."
                );

                let (out_chunk_decl, out_chunk_impl, aux_chunk) =
                    codegen::generate_fn(req, entity, None, &auxlib_name)?;
                ffi_stream.append_all(out_chunk_decl);
                out_stream.append_all(out_chunk_impl);
                aux_source.write_all(&aux_chunk)?;
                aux_source_required |= !aux_chunk.is_empty();
            }
            CxxForeignItem::Impl(req) => {
                let struct_ty = foreign_mod
                    .items
                    .iter()
                    .find_map(|x| match x {
                        CxxForeignItem::Struct(x) if &x.ident == req.self_ty.get_ident()? => {
                            Some(x)
                        }
                        _ => None,
                    })
                    .unwrap();
                let base_cxx_path = find_cxx_path(&struct_ty.attrs)
                    .map_or_else(|| Cow::Owned(struct_ty.ident.to_string()), Cow::Borrowed);

                let mut inner_out_stream = TokenStream::new();
                for item in &req.items {
                    let entity = analysis::find_fn(
                        &translation_unit,
                        &format!(
                            "{base_cxx_path}::{}",
                            find_cxx_path(&item.attrs).unwrap_or(&item.sig.ident.to_string())
                        ),
                        &item.sig,
                        &mappings,
                    )
                    .unwrap_or_else(|| panic!("Entity not found: {:#?}", req));

                    let (out_chunk_decl, out_chunk_impl, aux_chunk) =
                        codegen::generate_fn(item, entity, Some(&struct_ty.ident), &auxlib_name)?;
                    ffi_stream.append_all(out_chunk_decl);
                    inner_out_stream.append_all(out_chunk_impl);
                    aux_source.write_all(&aux_chunk)?;
                    aux_source_required |= !aux_chunk.is_empty();
                }

                let ident = &struct_ty.ident;
                out_stream.append_all(quote! {
                    impl #ident {
                        #inner_out_stream
                    }
                });
            }
            _ => {}
        }
    }

    // Build the auxiliary library if required.
    if aux_source_required {
        wrappers::build_auxiliary_library(auxlib_path, &aux_source_path)?;
    }

    ffi_stream.append_all(out_stream);
    Ok(ffi_stream)
}

fn find_cxx_path(attrs: &[CxxForeignAttr]) -> Option<&str> {
    attrs.iter().find_map(|x| match x {
        CxxForeignAttr::CxxPath(x) => Some(x.as_str()),
        _ => None,
    })
}
