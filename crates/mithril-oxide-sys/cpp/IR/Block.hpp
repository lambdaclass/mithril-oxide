#pragma once

#include <memory>

#include <mlir/IR/Block.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <rust/cxx.h>

#include "../lib.hpp"


namespace mithril_oxide_sys {

using mlir::Block;
using mlir::BlockArgument;
using mlir::Location;
using mlir::Type;


void Block_addArgument(Block &block, const void*type, const void* loc);
void* Block_getArgument(Block &block, unsigned i);

rust::String Block_print(Block &block);

} // namespace mithril_oxide_sys
