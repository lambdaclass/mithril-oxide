#include "BuiltinAttributes.hpp"

#include <memory>

#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

std::unique_ptr<StringAttr> StringAttr_get(MLIRContext &context, rust::Str value)
{
    return std::make_unique<StringAttr>(StringAttr::get(
        &context,
        mlir::StringRef(value.data(), value.length())
    ));
}

std::unique_ptr<IntegerAttr> IntegerAttr_get(MLIRContext &context, rust::Str value)
{
    auto int_val = llvm::APSInt(llvm::StringRef(value.data(), value.length()));
    return std::make_unique<IntegerAttr>(IntegerAttr::get(&context, int_val));
}

std::unique_ptr<BoolAttr> BoolAttr_get(MLIRContext &context, bool value)
{
    return std::make_unique<BoolAttr>(BoolAttr::get(&context, value));
}

std::unique_ptr<DenseElementsAttr> DenseElementsAttr_get(
    const ShapedType &type,
    rust::Slice<const Attribute *const> values
)
{
    std::vector<Attribute> values_vec;

    for (const auto &value : values)
        values_vec.push_back(*value);

    return std::make_unique<DenseElementsAttr>(DenseElementsAttr::get(type, values_vec));
}

#define MITHRIL_CAST_TO_ATTR_IMPL(FROM_TYPE) std::unique_ptr<Attribute> FROM_TYPE ## _to_Attribute(const FROM_TYPE &x) \
    { \
         return std::make_unique<Attribute>(x); \
    }

MITHRIL_CAST_TO_ATTR_IMPL(DictionaryAttr);
MITHRIL_CAST_TO_ATTR_IMPL(StringAttr);
MITHRIL_CAST_TO_ATTR_IMPL(FloatAttr);
MITHRIL_CAST_TO_ATTR_IMPL(IntegerAttr);
MITHRIL_CAST_TO_ATTR_IMPL(DenseElementsAttr);
MITHRIL_CAST_TO_ATTR_IMPL(DenseIntElementsAttr);
MITHRIL_CAST_TO_ATTR_IMPL(DenseFPElementsAttr);
MITHRIL_CAST_TO_ATTR_IMPL(BoolAttr);
MITHRIL_CAST_TO_ATTR_IMPL(FlatSymbolRefAttr);

} // namespace mithril_oxide_sys
