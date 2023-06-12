#include "BuiltinAttributes.hpp"

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys {

std::unique_ptr<StringAttr> StringAttr_get(MLIRContext &context)
{
        return std::make_unique<StringAttr>(StringAttr::get(&context));
}

} // namespace mithril_oxide_sys
