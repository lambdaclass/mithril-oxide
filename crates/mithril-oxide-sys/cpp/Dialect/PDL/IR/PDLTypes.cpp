#include "PDLTypes.hpp"

#include <mlir/Dialect/PDL/IR/PDLTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>


namespace mithril_oxide_sys::pdl {

const void *AttributeType_get(MLIRContext *ctx)
{
    return AttributeType::get(ctx).getAsOpaquePointer();
}

const void *OperationType_get(MLIRContext *ctx)
{
    return OperationType::get(ctx).getAsOpaquePointer();
}

const void *RangeType_get(const void *elementType)
{
    return RangeType::get(mlir::Type::getFromOpaquePointer(elementType)).getAsOpaquePointer();
}

const void *TypeType_get(MLIRContext *ctx)
{
    return TypeType::get(ctx).getAsOpaquePointer();
}

const void *ValueType_get(MLIRContext *ctx)
{
    return ValueType::get(ctx).getAsOpaquePointer();
}

} // namespace mithril_oxide_sys::pdl
