#include "Block.hpp"

namespace mithril_oxide_sys {

void Block_addArgument(Block &block, const Type& type, const Location& loc)
{
    block.addArgument(type, loc);
}

}
