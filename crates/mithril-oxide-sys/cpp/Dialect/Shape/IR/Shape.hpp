#pragma once

#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys::shape {

using mlir::MLIRContext;
using mlir::shape::ShapeType;
using mlir::shape::SizeType;
using mlir::shape::ValueShapeType;
using mlir::shape::WitnessType;


const void *ShapeType_get(MLIRContext *ctx);
const void *SizeType_get(MLIRContext *ctx);
const void *ValueShapeType_get(MLIRContext *ctx);
const void *WitnessType_get(MLIRContext *ctx);

} // namespace mithril_oxide_sys::shape
