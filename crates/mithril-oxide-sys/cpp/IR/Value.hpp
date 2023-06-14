#pragma once

#include <memory>

#include <mlir/IR/Block.h>
#include <mlir/IR/Value.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

using mlir::BlockArgument;
using mlir::Value;


std::unique_ptr<Value> BlockArgument_toValue(const BlockArgument& arg);

rust::String Value_print(Value &op);

} // namespace mithril_oxide_sys
