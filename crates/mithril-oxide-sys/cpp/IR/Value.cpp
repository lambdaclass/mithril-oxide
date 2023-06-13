#include "Value.hpp"

namespace mithril_oxide_sys
{

std::unique_ptr<Value> BlockArgument_toValue(const BlockArgument& arg)
{
    return std::make_unique<Value>(arg);
}

} // namespace mithril_oxide_sys
