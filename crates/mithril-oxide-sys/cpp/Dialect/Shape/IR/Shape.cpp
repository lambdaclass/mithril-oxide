#include "Shape.hpp"

#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys::shape {

const void *ShapeType_get(MLIRContext *ctx)
{
    return ShapeType::get(ctx);
}

const void *SizeType_get(MLIRContext *ctx)
{
    return SizeType::get(ctx);
}

const void *ValueShapeType_get(MLIRContext *ctx)
{
    return ValueShapeType::get(ctx);
}

const void *WitnessType_get(MLIRContext *ctx)
{
    return WitnessType::get(ctx);
}

} // namespace mithril_oxide_sys::shape
