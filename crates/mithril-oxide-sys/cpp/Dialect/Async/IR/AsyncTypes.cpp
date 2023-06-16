#include "AsyncTypes.hpp"

#include <mlir/Dialect/Async/IR/AsyncTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>


namespace mithril_oxide_sys::async {

const void *CoroHandleType_get(MLIRContext *ctx)
{
    return CoroHandleType::get(ctx).getAsOpaquePointer();
}

const void *CoroIdType_get(MLIRContext *ctx)
{
    return CoroIdType::get(ctx).getAsOpaquePointer();
}

const void *CoroStateType_get(MLIRContext *ctx)
{
    return CoroStateType::get(ctx).getAsOpaquePointer();
}

const void *GroupType_get(MLIRContext *ctx)
{
    return GroupType::get(ctx).getAsOpaquePointer();
}

const void *TokenType_get(MLIRContext *ctx)
{
    return TokenType::get(ctx).getAsOpaquePointer();
}

const void *ValueType_get(const void *type)
{
    return ValueType::get(mlir::Type::getFromOpaquePointer(type)).getAsOpaquePointer();
}

} // namespace mithril_oxide_sys::async
