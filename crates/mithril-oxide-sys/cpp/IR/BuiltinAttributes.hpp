#pragma once

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys {

using mlir::MLIRContext;
using mlir::StringAttr;


std::unique_ptr<StringAttr> StringAttr_get(MLIRContext &context);

} // namespace mithril_oxide_sys
