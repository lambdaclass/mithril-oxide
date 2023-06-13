#include "DialectRegistry.hpp"

#include <memory>

#include <mlir/IR/DialectRegistry.h>


namespace mithril_oxide_sys {

std::unique_ptr<DialectRegistry> DialectRegistry_new()
{
    return std::make_unique<DialectRegistry>();
}

} // namespace mithril_oxide_sys
