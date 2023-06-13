#pragma once

#include <memory>

#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys {

using mlir::MLIRContext;

std::unique_ptr<MLIRContext> MLIRContext_new();

} // namespace mithril_oxide_sys
