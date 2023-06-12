#include "BuiltinOps.hpp"

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

std::unique_ptr<ModuleOp> ModuleOp_create(const Location &loc, const rust::Str *name)
{
    return std::make_unique<ModuleOp>(ModuleOp::create(loc));
}

void ModuleOp_setSymNameAttr(ModuleOp &op, const StringAttr &value)
{
    op.setSymNameAttr(value);
}

void ModuleOp_setSymVisibilityAttr(ModuleOp &op, const StringAttr &value)
{
    op.setSymVisibilityAttr(value);
}

} // namespace mithril_oxide_sys
