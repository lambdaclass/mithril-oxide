#pragma once

#include <mlir/Dialect/Transform/IR/TransformTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys::transform {

using mlir::transform::AnyOpType;
using mlir::transform::OperationType;
using mlir::transform::ParamType;
using mlir::MLIRContext;


const void *AnyOpType_get(MLIRContext *ctx);
const void *OperationType_get(MLIRContext *ctx, rust::Str name);
const void *ParamType_get(MLIRContext *ctx, const void *type);

} // namespace mithril_oxide_sys::transform
