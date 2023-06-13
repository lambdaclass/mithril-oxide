#pragma once

#include <mlir/IR/Value.h>


namespace mithril_oxide_sys {

using mlir::Value;
using mlir::BlockArgument;

std::unique_ptr<Value> BlockArgument_toValue(const BlockArgument& arg);

} // namespace mithril_oxide_sys
