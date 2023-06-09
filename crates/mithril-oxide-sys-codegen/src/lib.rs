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

/// Generate a `TokenStream` with the C++ bindings and potentially an auxiliary static library from
/// an input `TokenStream`.
///
/// # Errors
///
/// > TODO: Why?
///
/// # Panics
///
/// > TODO: Why?
#[allow(clippy::too_many_lines)]
pub fn codegen(
    auxlib_path: &Path,
    stream: TokenStream,
) -> Result<TokenStream, Box<dyn std::error::Error>> {
    // Parse the custom `mod` item (see `parse` module).
    let foreign_mod: CxxForeignMod = syn::parse2(stream)?;

    // Initialize clang for later use.
    let clang = Clang::new()?;
    let index = Index::new(&clang, true, false);

    // Clang only parses files.
    let temp_dir = tempdir()?;
    let ast_source_path = temp_dir.path().join("ast.cpp");
    let aux_source_path = temp_dir.path().join("aux.cpp");

    // Write C++ source for analysis into `ast_source_path`.
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

    // Parse the C++ source for analysis.
    let translation_unit = analysis::parse_cpp(&index, &ast_source_path)?;

    // Initialize the output streams for both the Rust (bindings) and C++ (aux library) along with
    // a flag on whether the later is necessary.
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

    // Process types and generate their mappings. They are processed first because the funcitons and
    // methods may depend on them.
    let mut mappings = HashMap::new();
    for item in &foreign_mod.items {
        match item {
            CxxForeignItem::Enum(req) => {
                // Extract the C++ path (with namespace, etc) of the item, then search for the
                // matching C++ entity using clang.
                let cxx_path = find_cxx_path(&req.attrs)
                    .map_or_else(|| Cow::Owned(req.ident.to_string()), Cow::Borrowed);
                let entity =
                    analysis::find_enum(&translation_unit, &cxx_path).expect("Entity not found");

                // Register the mapping from Rust to C++ (see the function and method processing).
                mappings.insert(req.ident.clone(), cxx_path.to_string());

                // Generate the item code for both Rust and C++ (aux library) and append it to their
                // respective streams.
                let (out_chunk, aux_chunk) = codegen::generate_enum(req, entity)?;
                out_stream.append_all(out_chunk);
                aux_source.write_all(&aux_chunk)?;
            }
            CxxForeignItem::Struct(req) => {
                // Extract the C++ path (with namespace, etc) of the item, then search for the
                // matching C++ entity using clang.
                let cxx_path = find_cxx_path(&req.attrs)
                    .map_or_else(|| Cow::Owned(req.ident.to_string()), Cow::Borrowed);
                let entity =
                    analysis::find_struct(&translation_unit, &cxx_path).expect("Entity not found");

                // Register the mapping from Rust to C++ (see the function and method processing).
                mappings.insert(req.ident.clone(), cxx_path.to_string());

                // Generate the item code for both Rust and C++ (aux library) and append it to their
                // respective streams.
                let (out_chunk, aux_chunk) = codegen::generate_struct(req, entity)?;
                out_stream.append_all(out_chunk);
                aux_source.write_all(&aux_chunk)?;
            }
            _ => {}
        }
    }

    // Extract the auxiliary library name (for linking purposes, ex. `libauxlib.a` into `auxlib` to
    // be used like in `-lauxlib`).
    let auxlib_name = auxlib_path
        .with_extension("")
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
        .strip_prefix("lib")
        .unwrap()
        .to_string();

    // Process functions and methods using the precomputed mappings.
    let mut ffi_stream = out_stream;
    let mut out_stream = TokenStream::new();
    for item in &foreign_mod.items {
        match item {
            CxxForeignItem::Fn(req) => {
                // Search for the matching C++ entity using clang.
                let entity = analysis::find_fn(
                    &translation_unit,
                    find_cxx_path(&req.attrs).unwrap_or(&req.sig.ident.to_string()),
                    &req.sig,
                    &mappings,
                )
                .expect("Entity not found");

                // Free functions must not be methods.
                assert_eq!(
                    entity.get_kind(),
                    EntityKind::FunctionDecl,
                    "Non-impl methods (functions with self) are not allowed."
                );

                // Search for the matching C++ entity using clang.
                let (out_chunk_decl, out_chunk_impl, aux_chunk) =
                    codegen::generate_fn(req, entity, None, &auxlib_name)?;
                ffi_stream.append_all(out_chunk_decl);
                out_stream.append_all(out_chunk_impl);
                aux_source.write_all(&aux_chunk)?;
                aux_source_required |= !aux_chunk.is_empty();
            }
            CxxForeignItem::Impl(req) => {
                // Retrieve the struct type (or self type) from the generated items, then extract
                // the C++ path (with namespace, etc) of the item and the C++ path (with namespace,
                // etc).
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

                // Process each item into a separate TokenStream (Rust). The auxlib stream can be
                // the same.
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
                    .unwrap_or_else(|| {
                        panic!(
                            "Entity not found: {} (in {})",
                            find_cxx_path(&item.attrs).map_or_else(
                                || Cow::Owned(item.sig.ident.to_string()),
                                Cow::Borrowed
                            ),
                            find_cxx_path(&req.attrs).map_or_else(
                                || Cow::Owned(req.self_ty.get_ident().unwrap().to_string()),
                                Cow::Borrowed
                            )
                        )
                    });

                    let (out_chunk_decl, out_chunk_impl, aux_chunk) = codegen::generate_fn(
                        item,
                        entity,
                        Some((&struct_ty.ident, &mappings[&struct_ty.ident])),
                        &auxlib_name,
                    )?;
                    ffi_stream.append_all(out_chunk_decl);
                    inner_out_stream.append_all(out_chunk_impl);
                    aux_source.write_all(&aux_chunk)?;
                    aux_source_required |= !aux_chunk.is_empty();
                }

                // Encapsulate the items' TokenStream into an `impl { ... }` block and append it.
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

    // Revert the switching done earlier and merge both (extern decls & actual bindings) streams.
    ffi_stream.append_all(out_stream);
    Ok(ffi_stream)
}

/// Return the `cxx_path` attribute's value if present, or the type name if not.
fn find_cxx_path(attrs: &[CxxForeignAttr]) -> Option<&str> {
    attrs.iter().find_map(|x| match x {
        CxxForeignAttr::CxxPath(x) => Some(x.as_str()),
        _ => None,
    })
}
