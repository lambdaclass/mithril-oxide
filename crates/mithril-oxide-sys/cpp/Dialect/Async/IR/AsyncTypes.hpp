#pragma once

#include <mlir/Dialect/Async/IR/AsyncTypes.h>
#include <mlir/IR/MLIRContext.h>

#include "../../../lib.hpp"


namespace mithril_oxide_sys::async {

using mlir::async::CoroHandleType;
using mlir::async::CoroIdType;
using mlir::async::CoroStateType;
using mlir::async::GroupType;
using mlir::async::TokenType;
using mlir::async::ValueType;
using mlir::MLIRContext;


const void *CoroHandleType_get(MLIRContext *ctx);
const void *CoroIdType_get(MLIRContext *ctx);
const void *CoroStateType_get(MLIRContext *ctx);
const void *GroupType_get(MLIRContext *ctx);
const void *TokenType_get(MLIRContext *ctx);
const void *ValueType_get(const void *type);

} // namespace mithril_oxide_sys::async
