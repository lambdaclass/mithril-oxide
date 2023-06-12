#include "BuiltinAttributes.hpp"

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

std::unique_ptr<StringAttr> StringAttr_get(MLIRContext &context, const rust::Str *value)
{
    if (value == nullptr)
        return std::make_unique<StringAttr>(StringAttr::get(&context));
    else
        return std::make_unique<StringAttr>(StringAttr::get(
            &context,
            mlir::StringRef(value->data(), value->length())
        ));
}

} // namespace mithril_oxide_sys
