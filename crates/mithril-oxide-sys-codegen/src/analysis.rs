use clang::{Entity, EntityKind, EntityVisitResult, Index, TranslationUnit, TypeKind};
use std::{collections::HashMap, iter::Peekable, path::Path};
use syn::{FnArg, Ident, ReturnType, Signature, Type};

// TODO: Filter out private and protected APIs (they shouldn't be available from Rust).

/// Parse a C++ file into a clang translation unit.
pub fn parse_cpp<'c>(
    index: &'c Index,
    path: &Path,
) -> Result<TranslationUnit<'c>, Box<dyn std::error::Error>> {
    let translation_unit = index
        .parser(path)
        .arguments(&{
            let mut args = vec!["-std=c++17".to_string()];

            // Mac OS nonsense.
            #[cfg(target_os = "macos")]
            args.push("-isysroot/".to_string());

            args.extend(
                crate::wrappers::extract_clang_include_paths(path)?
                    .into_iter()
                    .map(|x| format!("-I{x}")),
            );

            args
        })
        .skip_function_bodies(true)
        .parse()?;

    let diagnostics = translation_unit.get_diagnostics();
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");

    Ok(translation_unit)
}

/// Search for an enum declaration in the C++ AST.
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

/// Search for a function declaration in the C++ AST.
///
/// Apart from the obvious stuff (same self type if applicable, path, return type and arguments),
/// the const specification (on methods) is also checked.
pub fn find_fn<'c>(
    translation_unit: &'c TranslationUnit,
    path: &str,
    sig: &Signature,
    mappings: &HashMap<Ident, String>,
) -> Option<Entity<'c>> {
    // Recursive helper function. Each invocation will walk a single AST level until it finds the
    // current path item.
    fn inner<'c, 'a>(
        entity: Entity<'c>,
        mut path: Peekable<impl Clone + Iterator<Item = &'a str>>,
        sig: &Signature,
        mappings: &HashMap<Ident, String>,
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
                    EntityKind::FunctionDecl
                    | EntityKind::Constructor
                    | EntityKind::Destructor
                    | EntityKind::Method => {
                        let is_method = sig
                            .inputs
                            .first()
                            .is_some_and(|x| matches!(x, FnArg::Receiver(_)));

                        let is_same_const = is_method
                            .then(|| match &sig.inputs[0] {
                                FnArg::Receiver(x) => {
                                    entity.is_const_method() == x.mutability.is_none()
                                }
                                FnArg::Typed(_) => panic!(),
                            })
                            .unwrap_or(true);

                        let is_same_output = match &sig.output {
                            ReturnType::Default => {
                                entity.get_result_type().unwrap().get_kind() == TypeKind::Void
                            }
                            ReturnType::Type(_, ty) => {
                                compare_types(ty, &entity.get_result_type().unwrap(), mappings)
                            }
                        };
                        let is_same_inputs = {
                            let args = entity.get_arguments().unwrap();

                            let len_matches =
                                args.len() + usize::from(is_method) == sig.inputs.len();
                            let arg_matches = sig
                                .inputs
                                .iter()
                                .skip(usize::from(is_method))
                                .zip(args)
                                .all(|(l, r)| {
                                    compare_types(
                                        match l {
                                            FnArg::Receiver(_) => unreachable!(),
                                            FnArg::Typed(x) => &x.ty,
                                        },
                                        &r.get_type().unwrap(),
                                        mappings,
                                    )
                                });

                            len_matches && arg_matches
                        };

                        if is_same_const && is_same_output && is_same_inputs {
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
                result = inner(entity, path.clone(), sig, mappings);
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
        mappings,
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

/// Compare a Rust type with its C++ counterpart.
fn compare_types(lhs: &Type, rhs: &clang::Type, mappings: &HashMap<Ident, String>) -> bool {
    match lhs {
        Type::Path(ty) => {
            assert!(ty.qself.is_none());

            let canonical_type = rhs.get_canonical_type();
            if ty.path.is_ident("Self") {
                true
            } else if ty.path.is_ident("bool") {
                canonical_type.get_kind() == TypeKind::Bool
            } else if ty.path.is_ident("u8")
                || ty.path.is_ident("u16")
                || ty.path.is_ident("u32")
                || ty.path.is_ident("u64")
            {
                let target_width = if ty.path.is_ident("u8") {
                    1
                } else if ty.path.is_ident("u16") {
                    2
                } else if ty.path.is_ident("u32") {
                    4
                } else {
                    8
                };

                canonical_type.is_unsigned_integer()
                    && canonical_type.get_sizeof().unwrap() == target_width
            } else if ty.path.is_ident("i8")
                || ty.path.is_ident("i16")
                || ty.path.is_ident("i32")
                || ty.path.is_ident("i64")
            {
                let target_width = if ty.path.is_ident("i8") {
                    1
                } else if ty.path.is_ident("i16") {
                    2
                } else if ty.path.is_ident("i32") {
                    4
                } else {
                    8
                };

                canonical_type.is_unsigned_integer()
                    && canonical_type.get_sizeof().unwrap() == target_width
            } else if ty.path.is_ident("f32") {
                canonical_type.get_kind() == TypeKind::Float
            } else if ty.path.is_ident("f64") {
                canonical_type.get_kind() == TypeKind::Double
            } else {
                let mapped_ty = &mappings[ty.path.get_ident().unwrap()];
                mapped_ty == &canonical_type.get_display_name()
            }
        }
        Type::Reference(ty) => match rhs.get_kind() {
            TypeKind::LValueReference => {
                ty.mutability.is_some() != rhs.is_const_qualified()
                    && compare_types(&ty.elem, &rhs.get_pointee_type().unwrap(), mappings)
            }
            _ => panic!(),
        },
        _ => todo!(),
    }
}
