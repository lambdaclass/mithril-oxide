#pragma once

#include <memory>

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

using mlir::Operation;
using mlir::OpResult;

using c_void = void;


rust::Str Operation_getName(Operation &op);

rust::String Operation_print(Operation &op);

void* Operation_getResult(Operation &op, unsigned idx);

} // namespace mithril_oxide_sys
