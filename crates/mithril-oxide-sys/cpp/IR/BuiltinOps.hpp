#pragma once

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>

#include "../lib.hpp"


namespace mithril_oxide_sys {

using mlir::ModuleOp;
using mlir::Location;
using mlir::StringAttr;


std::unique_ptr<ModuleOp> ModuleOp_create(const Location &loc);
// value - StringAttr
void ModuleOp_setSymNameAttr(ModuleOp &op, const void* value);
// value - StringAttr
void ModuleOp_setSymVisibilityAttr(ModuleOp &op, const void* value);

} // namespace mithril_oxide_sys
