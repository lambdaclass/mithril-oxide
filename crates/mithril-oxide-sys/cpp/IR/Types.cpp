#include "Types.hpp"
#include <memory>

namespace mithril_oxide_sys
{

std::unique_ptr<Type> Value_getType(const Value& value)
{
    return std::make_unique<Type>(value.getType());
}

} // namespace mithril_oxide_sys
