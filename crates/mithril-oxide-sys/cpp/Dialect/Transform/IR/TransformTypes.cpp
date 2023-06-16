#include "TransformTypes.hpp"

#include <mlir/Dialect/Transform/IR/TransformTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys::transform {

const void *AnyOpType_get(MLIRContext *ctx)
{
    return AnyOpType::get(ctx).getAsOpaquePointer();
}

const void *OperationType_get(MLIRContext *ctx, rust::Str name)
{
    return OperationType::get(
        ctx,
        mlir::StringRef(name.data(), name.length())
    ).getAsOpaquePointer();
}

const void *ParamType_get(MLIRContext *ctx, const void *type)
{
    return ParamType::get(ctx, mlir::Type::getFromOpaquePointer(type)).getAsOpaquePointer();
}

} // namespace mithril_oxide_sys::transform
