#pragma once

#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>


namespace mithril_oxide_sys {

using mlir::Type;
using mlir::Value;

std::unique_ptr<Type> Value_getType(const Value& value);

} // namespace mithril_oxide_sys
