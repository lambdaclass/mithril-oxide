#pragma once

#include "Location.hpp"
#include <memory>

#include <mlir/IR/Block.h>
#include <mlir/IR/BlockSupport.h>

#include <rust/cxx.h>


namespace mithril_oxide_sys {

using mlir::Block;
using mlir::Type;
using mlir::Location;

void Block_addArgument(Block &block, const Type& type, const Location& loc);

} // namespace mithril_oxide_sys
