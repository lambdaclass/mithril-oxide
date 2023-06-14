#include "Block.hpp"

#include <memory>

#include <mlir/IR/Block.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>


namespace mithril_oxide_sys {

void Block_addArgument(Block &block, const Type& type, const Location& loc)
{
    block.addArgument(type, loc);
}

std::unique_ptr<BlockArgument> Block_getArgument(Block &block, unsigned i)
{
    return std::make_unique<BlockArgument>(block.getArgument(i));
}

} // namespace mithril_oxide_sys
