#pragma once

#include <mlir/Dialect/PDL/IR/PDLTypes.h>
#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys::pdl {

using mlir::pdl::AttributeType;
using mlir::pdl::OperationType;
using mlir::pdl::RangeType;
using mlir::pdl::TypeType;
using mlir::pdl::ValueType;
using mlir::MLIRContext;


const void *AttributeType_get(MLIRContext *ctx);
const void *OperationType_get(MLIRContext *ctx);
const void *RangeType_get(MLIRContext *ctx);
const void *TypeType_get(MLIRContext *ctx);
const void *ValueType_get(MLIRContext *ctx);

} // namespace mithril_oxide_sys::pdl
