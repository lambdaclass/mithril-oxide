#pragma once

#include <memory>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <rust/cxx.h>
#include <string>


namespace mithril_oxide_sys {

using mlir::Operation;
using mlir::OpResult;


rust::Str Operation_getName(Operation &op);

rust::String Operation_print(Operation &op);

std::unique_ptr<OpResult> Operation_getResult(Operation &op, unsigned idx);

} // namespace mithril_oxide_sys
