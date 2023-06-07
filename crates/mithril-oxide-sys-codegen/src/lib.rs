#![feature(iterator_try_collect)]

use crate::parsing::{CxxForeignAttr, CxxForeignItem, CxxForeignMod};
use clang::{Clang, EntityKind, Index};
use proc_macro2::TokenStream;
use quote::TokenStreamExt;
use std::{
    fs::File,
    io::{self, Write},
};
use tempfile::tempdir;

mod analysis;
mod codegen;
mod parsing;
mod wrappers;

pub fn codegen(stream: TokenStream) -> Result<TokenStream, Box<dyn std::error::Error>> {
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

    let translation_unit = analysis::parse_cpp(&index, &ast_source_path);

    let mut out_stream = TokenStream::new();
    let mut aux_source = File::create(aux_source_path)?;
    let mut aux_source_required = false;

    // Insert include statements into the aux library.
    for item in &foreign_mod.items {
        if let CxxForeignItem::IncludeAttr(file) = item {
            writeln!(aux_source, "#include <{file}>")?;
        }
    }

    for item in &foreign_mod.items {
        match item {
            CxxForeignItem::Enum(req) => {
                let entity = analysis::find_enum(
                    &translation_unit,
                    find_cxx_path(&req.attrs).unwrap_or(&req.ident.to_string()),
                )
                .expect("Entity not found");

                let (out_chunk, aux_chunk) = codegen::generate_enum(req, entity)?;
                out_stream.append_all(out_chunk);
                aux_source.write_all(&aux_chunk)?;
            }
            CxxForeignItem::Fn(req) => {
                let entity = analysis::find_fn(
                    &translation_unit,
                    find_cxx_path(&req.attrs).unwrap_or(&req.sig.ident.to_string()),
                )
                .expect("Entity not found");
                assert_eq!(
                    entity.get_kind(),
                    EntityKind::FunctionDecl,
                    "Non-impl methods are not allowed."
                );

                let (out_chunk, aux_chunk) = codegen::generate_fn(req, entity)?;
                out_stream.append_all(out_chunk);
                aux_source.write_all(&aux_chunk)?;
            }
            CxxForeignItem::Impl(req) => {
                // let struct_ty = foreign_mod.items.iter().find_map(|x| match x {
                //     CxxForeignItem::Struct(x) if &x.ident == req.self_ty.get_ident().unwrap() => {
                //         Some(x)
                //     }
                //     _ => None,
                // });

                todo!()
            }
            CxxForeignItem::Struct(req) => {
                let entity = analysis::find_struct(
                    &translation_unit,
                    find_cxx_path(&req.attrs).unwrap_or(&req.ident.to_string()),
                )
                .expect("Entity not found");

                let (out_chunk, aux_chunk) = codegen::generate_struct(req, entity)?;
                out_stream.append_all(out_chunk);
                aux_source.write_all(&aux_chunk)?;
            }
            CxxForeignItem::IncludeAttr(_) => {}
        }
    }

    todo!()
}

fn find_cxx_path<'a>(attrs: &'a [CxxForeignAttr]) -> Option<&'a str> {
    attrs.iter().find_map(|x| match x {
        CxxForeignAttr::CxxPath(x) => Some(x.as_str()),
        _ => None,
    })
}
