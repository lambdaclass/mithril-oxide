use clang::{Entity, EntityKind, EntityVisitResult, Index, TranslationUnit, TypeKind};
use std::{iter::Peekable, path::Path};
use syn::{FnArg, ReturnType, Signature, Type};

pub fn parse_cpp<'c>(index: &'c Index, path: &Path) -> TranslationUnit<'c> {
    let translation_unit = index
        .parser(path)
        .arguments(&{
            let mut args = vec!["-std=c++17".to_string()];

            args.extend(
                crate::wrappers::extract_clang_include_paths(path)
                    .into_iter()
                    .map(|x| format!("-I{x}")),
            );

            args
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
            // Compare item names. If they differ, skip them.
            if entity.get_name().as_deref() != Some(path_segment) {
                return EntityVisitResult::Continue;
            }

            // Check if it's the last item in the path (i.e. what we are searching for) and return
            // it if it's a match.
            if path.peek().is_none() {
                return match entity.get_kind() {
                    EntityKind::EnumDecl => {
                        result = Some(entity);
                        EntityVisitResult::Break
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

    inner(translation_unit.get_entity(), path.split("::").peekable())
}

pub fn find_fn<'c>(
    translation_unit: &'c TranslationUnit,
    path: &str,
    sig: &Signature,
) -> Option<Entity<'c>> {
    // Recursive helper function. Each invocation will walk a single AST level until it finds the
    // current path item.
    fn inner<'c, 'a>(
        entity: Entity<'c>,
        mut path: Peekable<impl Clone + Iterator<Item = &'a str>>,
        sig: &Signature,
    ) -> Option<Entity<'c>> {
        let mut result = None;

        let path_segment = path.next().unwrap();
        entity.visit_children(|entity, _| {
            // Compare item names. If they differ, skip them.
            if entity.get_name().as_deref() != Some(path_segment) {
                return EntityVisitResult::Continue;
            }

            // Check if it's the last item in the path (i.e. what we are searching for) and return
            // it if it's a match.
            if path.peek().is_none() {
                return match entity.get_kind() {
                    EntityKind::FunctionDecl | EntityKind::Method => {
                        // TODO: Compare arguments (overloading compatibility) and return values.
                        let is_method = sig
                            .inputs
                            .first()
                            .is_some_and(|x| matches!(x, FnArg::Receiver(_)));

                        let is_same_output = match &sig.output {
                            ReturnType::Default => {
                                entity.get_result_type().unwrap().get_kind() == TypeKind::Void
                            }
                            ReturnType::Type(_, ty) => {
                                compare_types(ty, &entity.get_result_type().unwrap())
                            }
                        };
                        let is_same_inputs = {
                            let args = entity.get_arguments().unwrap();

                            let len_matches = args.len() + is_method as usize == sig.inputs.len();
                            let arg_matches =
                                sig.inputs
                                    .iter()
                                    .skip(is_method as _)
                                    .zip(args)
                                    .all(|(l, r)| {
                                        compare_types(
                                            match l {
                                                FnArg::Receiver(_) => unreachable!(),
                                                FnArg::Typed(x) => &x.ty,
                                            },
                                            &r.get_type().unwrap(),
                                        )
                                    });

                            len_matches && arg_matches
                        };

                        if is_same_output && is_same_inputs {
                            result = Some(entity);
                            EntityVisitResult::Break
                        } else {
                            EntityVisitResult::Continue
                        }
                    }
                    _ => EntityVisitResult::Continue,
                };
            }

            // If not the last item in the path, check if it is a container and recurse if so.
            if matches!(
                entity.get_kind(),
                EntityKind::Namespace | EntityKind::ClassDecl | EntityKind::StructDecl
            ) {
                result = inner(entity, path.clone(), sig);
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
        translation_unit.get_entity(),
        path.split("::").peekable(),
        sig,
    )
}

pub fn find_struct<'c>(translation_unit: &'c TranslationUnit, path: &str) -> Option<Entity<'c>> {
    // Recursive helper function. Each invocation will walk a single AST level until it finds the
    // current path item.
    fn inner<'c, 'a>(
        entity: Entity<'c>,
        mut path: Peekable<impl Clone + Iterator<Item = &'a str>>,
    ) -> Option<Entity<'c>> {
        let mut result = None;

        let path_segment = path.next().unwrap();
        entity.visit_children(|entity, _| {
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

    inner(translation_unit.get_entity(), path.split("::").peekable())
}

fn compare_types(lhs: &Type, rhs: &clang::Type) -> bool {
    match lhs {
        Type::Array(_) => todo!(),
        Type::BareFn(_) => todo!(),
        Type::Group(_) => todo!(),
        Type::ImplTrait(_) => todo!(),
        Type::Infer(_) => todo!(),
        Type::Macro(_) => todo!(),
        Type::Never(_) => todo!(),
        Type::Paren(_) => todo!(),
        Type::Path(ty) => todo!("{ty:?}, {rhs:?}"),
        Type::Ptr(_) => todo!(),
        Type::Reference(ty) => match rhs.get_kind() {
            TypeKind::LValueReference => {
                ty.mutability.is_some() != rhs.is_const_qualified()
                    && compare_types(&ty.elem, &rhs.get_canonical_type())
            }
            _ => panic!(),
        },
        Type::Slice(_) => todo!(),
        Type::TraitObject(_) => todo!(),
        Type::Tuple(_) => todo!(),
        Type::Verbatim(_) => todo!(),
        _ => unreachable!(),
    }
}
