#include "BuiltinOps.hpp"

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>


namespace mithril_oxide_sys {

std::unique_ptr<ModuleOp> ModuleOp_create(const void * loc)
{
    return std::make_unique<ModuleOp>(ModuleOp::create(Location::getFromOpaquePointer(loc)));
}

void ModuleOp_setSymNameAttr(ModuleOp &op, const void* value)
{
    op.setSymNameAttr(StringAttr::getFromOpaquePointer(value));
}

void ModuleOp_setSymVisibilityAttr(ModuleOp &op, const void* value)
{
    op.setSymVisibilityAttr(StringAttr::getFromOpaquePointer(value));
}

} // namespace mithril_oxide_sys
