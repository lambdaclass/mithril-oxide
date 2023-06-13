#pragma once

#include <memory>

#include <mlir/IR/Operation.h>
#include <rust/cxx.h>
#include <string>


namespace mithril_oxide_sys {

using mlir::Operation;


rust::Str Operation_getName(Operation &op);

rust::String Operation_print(Operation &op);

} // namespace mithril_oxide_sys
