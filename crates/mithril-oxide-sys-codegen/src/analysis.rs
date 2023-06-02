use crate::request::{
    RequestConstructor, RequestEnum, RequestItem, RequestMethod, RequestMethodImpl, RequestMod,
    RequestStruct,
};
use clang::{Entity, EntityKind, EntityVisitResult, Index, TranslationUnit, Type, TypeKind};
use std::{collections::HashMap, fs};
use syn::ReturnType;
use tempfile::tempdir;

pub fn load_cpp<'a>(index: &'a Index<'a>, source_code: &str) -> TranslationUnit<'a> {
    let dir = tempdir().unwrap();
    let path = dir.path().join("source.cpp");
    fs::write(&path, source_code).unwrap();

    let translation_unit = index
        .parser(path)
        .arguments(&[
            // Esteve's flags.
            "-D__cplusplus=201703L",
            "-I/usr/bin/../lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12",
            "-I/usr/bin/../lib/gcc/x86_64-linux-gnu/12/../../../../include/x86_64-linux-gnu/c++/12",
            "-I/usr/bin/../lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/backward",
            "-I/usr/lib/llvm-16/lib/clang/16/include",
            "-I/usr/local/include",
            "-I/usr/include/x86_64-linux-gnu",
            "-I/usr/include",
            "-I/usr/lib/llvm-16/include",
            // Edgar's flags.
            "-std=c++17",
            "-I/usr/include/x86_64-pc-linux-gnu",
            "-I/usr/lib/gcc/x86_64-pc-linux-gnu/12/include",
            "-I/usr/lib/gcc/x86_64-pc-linux-gnu/12/include/g++-v12",
            "-I/usr/local/include",
            "-I/usr/include",
            "-I/home/edgar/data/work/cairo_sierra_2_MLIR/llvm/dist/lib/clang/16/include",
            "-I/home/edgar/data/work/cairo_sierra_2_MLIR/llvm/dist/include",
        ])
        .skip_function_bodies(true)
        .parse()
        .unwrap();

    translation_unit
}

pub fn analyze_cpp<'a>(
    translation_unit: &'a TranslationUnit<'a>,
    request: &RequestMod,
) -> HashMap<String, MappedItemWithWithMethods<'a>> {
    // I have:
    //   - Types (syn) & methods (syn)
    //
    // I want:
    //   - Types (clang) & methods (clang)
    //
    // Methods are search by their name and arguments (overloads), therefore I need all types
    // matched before searching the methods.
    //
    // I need a function to check whether a method's arguments match those from syn.

    let entity = translation_unit.get_entity();

    // Find mappings from syn types to clang ones.
    let mappings = request
        .items
        .iter()
        .map(|item| match item {
            RequestItem::Struct(syn_struct) => {
                let clang_struct = find_struct(&entity, &syn_struct.path).unwrap();
                (
                    syn_struct.name.clone(),
                    MappedItem::Struct(syn_struct.clone(), clang_struct),
                )
            }
            RequestItem::Enum(syn_enum) => {
                let clang_enum = find_enum(&entity, &syn_enum.path).unwrap();
                (
                    syn_enum.name.clone(),
                    MappedItem::Enum(syn_enum.clone(), clang_enum),
                )
            }
        })
        .collect::<HashMap<_, _>>();

    // Find mappings from syn methods to clang ones.
    let mappings = mappings
        .into_iter()
        .map(|(name, mapped_item)| match mapped_item {
            MappedItem::Struct(syn_struct, clang_struct) => {
                let methods = syn_struct
                    .items
                    .iter()
                    .map(move |x| match x {
                        RequestMethodImpl::Constructor(syn_method) => {
                            find_constructor(&clang_struct, syn_method).unwrap()
                        }
                        RequestMethodImpl::Method(syn_method) => {
                            find_method(&clang_struct, syn_method).unwrap()
                        }
                    })
                    .collect::<Vec<_>>();

                (
                    name,
                    MappedItemWithWithMethods::Struct(syn_struct, clang_struct, methods),
                )
            }
            MappedItem::Enum(syn_enum, clang_enum) => {
                let variants = syn_enum
                    .variants
                    .iter()
                    .map(|x| find_variant(&clang_enum, x).unwrap())
                    .collect::<Vec<_>>();

                (
                    name,
                    MappedItemWithWithMethods::Enum(syn_enum, clang_enum, variants),
                )
            }
        })
        .collect::<HashMap<_, _>>();

    mappings
}

fn find_struct<'a>(entity: &Entity<'a>, path: &str) -> Option<Entity<'a>> {
    fn inner<'a>(entity: &Entity<'a>, path: &[&str]) -> Option<Entity<'a>> {
        let mut result = None;
        entity.visit_children(|entity, _| {
            let name_matches = entity.get_name().as_deref() == Some(path[0]);
            if path.len() == 1 {
                let kind_matches = matches!(
                    entity.get_kind(),
                    EntityKind::ClassDecl | EntityKind::StructDecl
                );

                if kind_matches && name_matches {
                    result = Some(entity);
                }
            } else {
                let kind_matches = matches!(
                    entity.get_kind(),
                    EntityKind::ClassDecl | EntityKind::StructDecl | EntityKind::Namespace
                );

                if kind_matches && name_matches {
                    result = inner(&entity, &path[1..]);
                }
            }

            if result.is_some() {
                EntityVisitResult::Break
            } else {
                EntityVisitResult::Continue
            }
        });

        result
    }

    let path = path.split("::").collect::<Vec<_>>();
    assert!(!path.is_empty());

    inner(entity, &path)
}

fn find_enum<'a>(entity: &Entity<'a>, path: &str) -> Option<Entity<'a>> {
    fn inner<'a>(entity: &Entity<'a>, path: &[&str]) -> Option<Entity<'a>> {
        let mut result = None;
        entity.visit_children(|entity, _| {
            let name_matches = entity.get_name().as_deref() == Some(path[0]);
            if path.len() == 1 {
                let kind_matches = matches!(entity.get_kind(), EntityKind::EnumDecl);

                if kind_matches && name_matches {
                    result = Some(entity);
                }
            } else {
                let kind_matches = matches!(
                    entity.get_kind(),
                    EntityKind::ClassDecl | EntityKind::StructDecl | EntityKind::Namespace
                );

                if kind_matches && name_matches {
                    result = inner(&entity, &path[1..]);
                }
            }

            if result.is_some() {
                EntityVisitResult::Break
            } else {
                EntityVisitResult::Continue
            }
        });

        result
    }

    let path = path.split("::").collect::<Vec<_>>();
    assert!(!path.is_empty());

    inner(entity, &path)
}

fn find_variant<'a>(entity: &Entity<'a>, name: &str) -> Option<Entity<'a>> {
    let mut result = None;
    entity.visit_children(|entity, _| match entity.get_kind() {
        EntityKind::EnumConstantDecl if entity.get_display_name().unwrap() == name => {
            result = Some(entity);
            EntityVisitResult::Break
        }
        _ => EntityVisitResult::Recurse,
    });

    result
}

fn find_constructor<'a>(entity: &Entity<'a>, request: &RequestConstructor) -> Option<Entity<'a>> {
    let mut result = None;
    entity.visit_children(|entity, _| {
        match entity.get_kind() {
            EntityKind::Constructor => {
                let args_match = entity
                    .get_type()
                    .unwrap()
                    .get_argument_types()
                    .unwrap()
                    .iter()
                    .zip(&request.args)
                    .all(|(clang_arg, (_, syn_arg))| type_matches(syn_arg, clang_arg));

                if args_match {
                    result = Some(entity);
                    return EntityVisitResult::Break;
                }
            }
            _ => {}
        }

        EntityVisitResult::Recurse
    });

    result
}

fn find_method<'a>(entity: &Entity<'a>, request: &RequestMethod) -> Option<Entity<'a>> {
    let mut result = None;
    entity.visit_children(|entity, _| {
        match entity.get_kind() {
            EntityKind::Method if entity.get_name().unwrap() == request.name => {
                let ty = entity.get_type().unwrap();

                let ret_matches = match &request.ret {
                    ReturnType::Default => {
                        ty.get_result_type().unwrap().get_kind() == TypeKind::Void
                    }
                    ReturnType::Type(_, syn_arg) => {
                        type_matches(syn_arg, &ty.get_result_type().unwrap())
                    }
                };
                let args_match = entity
                    .get_type()
                    .unwrap()
                    .get_argument_types()
                    .unwrap()
                    .iter()
                    .zip(&request.args)
                    .all(|(clang_arg, (_, syn_arg))| type_matches(syn_arg, clang_arg));

                if ret_matches && args_match {
                    result = Some(entity);
                    return EntityVisitResult::Break;
                }
            }
            _ => {}
        }

        EntityVisitResult::Recurse
    });

    result
}

fn type_matches(syn_arg: &syn::Type, clang_arg: &Type) -> bool {
    match clang_arg.get_kind() {
        TypeKind::Bool => syn_arg == &syn::parse_quote!(bool),
        TypeKind::Elaborated => type_matches(syn_arg, &clang_arg.get_elaborated_type().unwrap()),
        TypeKind::Enum => {
            // TODO: Investigate potential bug when there's a naming collision between
            //   clang_arg.get_display_name() and syn_arg.
            syn_arg
                == &syn::parse_str(
                    &clang_arg
                        .get_declaration()
                        .unwrap()
                        .get_display_name()
                        .unwrap(),
                )
                .unwrap()
        }
        x => todo!("type {x:?} not implemented"),
    }
}

#[derive(Debug)]
pub enum MappedItem<'a> {
    Struct(RequestStruct, Entity<'a>),
    Enum(RequestEnum, Entity<'a>),
}

#[derive(Debug)]
pub enum MappedItemWithWithMethods<'a> {
    Struct(RequestStruct, Entity<'a>, Vec<Entity<'a>>),
    Enum(RequestEnum, Entity<'a>, Vec<Entity<'a>>),
}
