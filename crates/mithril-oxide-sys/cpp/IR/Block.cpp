#include "Block.hpp"

#include <memory>

#include <mlir/IR/Block.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

void Block_addArgument(Block &block, const void*type, const Location& loc)
{
    block.addArgument(Type::getFromOpaquePointer(type), loc);
}

void* Block_getArgument(Block &block, unsigned i)
{
    return block.getArgument(i).getAsOpaquePointer();
}

} // namespace mithril_oxide_sys
