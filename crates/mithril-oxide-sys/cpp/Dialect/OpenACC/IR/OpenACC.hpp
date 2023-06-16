#pragma once

#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys::acc {

using mlir::MLIRContext;


void *DataBoundsType_create(MLIRContext *ctx);

} // namespace mithril_oxide_sys::acc
