use clang::{Entity, EntityKind, EntityVisitResult, Index, TranslationUnit};
use std::{iter::Peekable, path::Path};

pub fn parse_cpp<'c>(index: &'c Index, path: &Path) -> TranslationUnit<'c> {
    let translation_unit = index
        .parser(path)
        .arguments(&{
            let mut args = vec![
                "-std=c++17".to_string(),
            ];

            args.extend(
                crate::wrappers::extract_clang_include_paths(path)
                    .into_iter()
                    .map(|x| format!("-I{x}")),
            );

            dbg!(args)
        })
        .skip_function_bodies(true)
        .parse()
        .unwrap();

    let diagnostics = translation_unit.get_diagnostics();
    if !diagnostics.is_empty() {
        panic!("{diagnostics:#?}");
    }

    translation_unit
}

pub fn find_enum<'c>(translation_unit: &'c TranslationUnit, path: &str) -> Option<Entity<'c>> {
    // Recursive helper function. Each invocation will walk a single AST level until it finds the
    // current path item.
    fn inner<'c, 'a>(
        entity: Entity<'c>,
        mut path: Peekable<impl Clone + Iterator<Item = &'a str>>,
    ) -> Option<Entity<'c>> {
        let mut result = None;

        let path_segment = path.next().unwrap();
        entity.visit_children(|entity, _| {
            let entity = entity.get_canonical_entity();

            // Compare item names. If they differ, skip them.
            if entity.get_name().as_deref() != Some(path_segment) {
                return EntityVisitResult::Continue;
            }

    // Find mappings from syn methods to clang ones.
    mappings
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
                            find_method(&clang_struct, syn_method)
                                .expect(&format!("couldn't find method for {:?}", syn_method))
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
            MappedItem::Function(syn_func, clang_func) => (
                name,
                MappedItemWithWithMethods::Function(syn_func, clang_func, vec![]),
            ),
        })
        .collect::<HashMap<_, _>>()
}

fn find_function<'a>(entity: &Entity<'a>, request: &RequestFunction) -> Option<Entity<'a>> {
    let mut result = None;
    entity.visit_children(|entity, _| {
        match entity.get_kind() {
            EntityKind::FunctionDecl if entity.get_name().unwrap() == request.cxx_ident => {
                let ty = entity.get_type().unwrap();
                let ret_matches = match &request.ret {
                    ReturnType::Default => {
                        ty.get_result_type().unwrap().get_kind() == TypeKind::Void
                    }
                    _ => EntityVisitResult::Continue,
                };
            }

            // If not the last item in the path, check if it is a container and recurse if so.
            if matches!(
                entity.get_kind(),
                EntityKind::Namespace | EntityKind::ClassDecl | EntityKind::StructDecl
            ) {
                result = inner(entity, path.clone());
                if result.is_some() {
                    return EntityVisitResult::Break;
                }
            }

            // Otherwise just continue walking the current level.
            EntityVisitResult::Continue
        });

        result
    }

    inner(
        translation_unit.get_entity().get_canonical_entity(),
        path.split("::").peekable(),
    )
}

pub fn find_fn<'c>(translation_unit: &'c TranslationUnit, path: &str) -> Option<Entity<'c>> {
    // Recursive helper function. Each invocation will walk a single AST level until it finds the
    // current path item.
    fn inner<'c, 'a>(
        entity: Entity<'c>,
        mut path: Peekable<impl Clone + Iterator<Item = &'a str>>,
    ) -> Option<Entity<'c>> {
        let mut result = None;

        let path_segment = path.next().unwrap();
        entity.visit_children(|entity, _| {
            let entity = entity.get_canonical_entity();

            // Compare item names. If they differ, skip them.
            if entity.get_name().as_deref() != Some(path_segment) {
                return EntityVisitResult::Continue;
            }

            // Check if it's the last item in the path (i.e. what we are searching for) and return
            // it if it's a match.
            if path.peek().is_none() {
                return match entity.get_kind() {
                    EntityKind::FunctionDecl | EntityKind::Method => {
                        result = Some(entity);
                        EntityVisitResult::Break
                    }
                    _ => EntityVisitResult::Continue,
                };
            }

fn find_constructor<'a>(entity: &Entity<'a>, request: &RequestConstructor) -> Option<Entity<'a>> {
    let mut result = None;

    entity.visit_children(|entity, _| {
        if let EntityKind::Constructor = entity.get_kind() {
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
        EntityVisitResult::Recurse
    });

        result
    }

fn find_method<'a>(struct_entity: &Entity<'a>, request: &RequestMethod) -> Option<Entity<'a>> {
    let mut result = None;
    struct_entity.visit_children(|entity, _| {
        match entity.get_kind() {
            EntityKind::Method if entity.get_name().unwrap() == request.name => {
                let ty = entity.get_type().unwrap();

            // Compare item names. If they differ, skip them.
            if entity.get_name().as_deref() != Some(path_segment) {
                return EntityVisitResult::Continue;
            }

            // Check if it's the last item in the path (i.e. what we are searching for) and return
            // it if it's a match.
            if path.peek().is_none() {
                return match entity.get_kind() {
                    EntityKind::ClassDecl | EntityKind::StructDecl => {
                        result = Some(entity);
                        EntityVisitResult::Break
                    }
                    _ => EntityVisitResult::Continue,
                };
                let args_match = entity
                    .get_type()
                    .unwrap()
                    .get_argument_types()
                    .unwrap()
                    .iter()
                    .zip(request.args.iter().skip(1)) // skip self on the rust side. todo: check self is correct?
                    .all(|(clang_arg, (_, syn_arg))| type_matches(syn_arg, clang_arg));

            // If not the last item in the path, check if it is a container and recurse if so.
            if matches!(
                entity.get_kind(),
                EntityKind::Namespace | EntityKind::ClassDecl | EntityKind::StructDecl
            ) {
                result = inner(entity, path.clone());
                if result.is_some() {
                    return EntityVisitResult::Break;
                }
            }

            // Otherwise just continue walking the current level.
            EntityVisitResult::Continue
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
        TypeKind::Void => syn_arg == &syn::parse_quote!(()),
        TypeKind::LValueReference => {
            if let syn::Type::Reference(type_ref) = syn_arg {
                let is_mut = type_ref.mutability.is_some();
                if is_mut != clang_arg.is_const_qualified() {
                    let name = clang_arg.get_display_name();
                    let clang_name = name.strip_suffix("&").unwrap().trim();
                    if let syn::Type::Path(p) = &*type_ref.elem {
                        if p.path.is_ident(&clang_name) {
                            return true;
                        }
                    }
                }
            }
            false
        }
        TypeKind::UInt => {
            if let syn::Type::Path(p) = syn_arg {
                match clang_arg.get_sizeof().unwrap() {
                    1 => p.path.is_ident("u8"),
                    2 => p.path.is_ident("u16"),
                    4 => p.path.is_ident("u32"),
                    8 => p.path.is_ident("u64"),
                    _ => unreachable!(),
                }
            } else {
                false
            }
        }
        TypeKind::Int => {
            if let syn::Type::Path(p) = syn_arg {
                match clang_arg.get_sizeof().unwrap() {
                    1 => p.path.is_ident("i8"),
                    2 => p.path.is_ident("i16"),
                    4 => p.path.is_ident("i32"),
                    8 => p.path.is_ident("i64"),
                    _ => unreachable!(),
                }
            } else {
                false
            }
        }
        x => todo!("type {x:?} not implemented"),
    }
}

#[derive(Debug)]
pub enum MappedItem<'a> {
    Struct(RequestStruct, Entity<'a>),
    Enum(RequestEnum, Entity<'a>),
    Function(RequestFunction, Entity<'a>),
}

#[derive(Debug)]
pub enum MappedItemWithWithMethods<'a> {
    Struct(RequestStruct, Entity<'a>, Vec<Entity<'a>>),
    Enum(RequestEnum, Entity<'a>, Vec<Entity<'a>>),
    Function(RequestFunction, Entity<'a>, Vec<Entity<'a>>),
}
