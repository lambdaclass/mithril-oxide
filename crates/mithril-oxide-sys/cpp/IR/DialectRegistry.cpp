#include "DialectRegistry.hpp"

#include <mlir/IR/DialectRegistry.h>
#include <memory>


namespace mithril_oxide_sys {

std::unique_ptr<DialectRegistry> DialectRegistry_new()
{
    return std::make_unique<DialectRegistry>();
}

} // namespace mithril_oxide_sys
