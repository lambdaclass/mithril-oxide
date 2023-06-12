#pragma once

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

using mlir::ModuleOp;
using mlir::Location;
using mlir::StringAttr;


std::unique_ptr<ModuleOp> ModuleOp_create(const Location &loc, const rust::Str *name);
void ModuleOp_setSymNameAttr(ModuleOp &op, const StringAttr &value);
void ModuleOp_setSymVisibilityAttr(ModuleOp &op, const StringAttr &value);

} // namespace mithril_oxide_sys
