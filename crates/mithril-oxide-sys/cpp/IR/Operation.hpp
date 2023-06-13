#pragma once

#include <memory>

#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>

#include <rust/cxx.h>


namespace mithril_oxide_sys {

using mlir::Operation;


rust::Str Operation_getName(Operation &op);



} // namespace mithril_oxide_sys
