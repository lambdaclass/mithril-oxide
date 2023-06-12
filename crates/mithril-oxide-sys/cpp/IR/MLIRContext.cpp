#include "MLIRContext.hpp"

#include <memory>

#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys {

std::unique_ptr<MLIRContext> MLIRContext_new()
{
    return std::make_unique<MLIRContext>();
}

} // namespace mithril_oxide_sys
