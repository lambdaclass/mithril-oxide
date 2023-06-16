#include "BuiltinAttributes.hpp"

#include <memory>

#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

const void* StringAttr_get(MLIRContext &context, rust::Str value)
{
    return StringAttr::get(
        &context,
        mlir::StringRef(value.data(), value.length())
    ).getAsOpaquePointer();
}

const void* FlatSymbolRefAttr_get(MLIRContext &context, rust::Str value)
{
    return FlatSymbolRefAttr::get(
        &context,
        mlir::StringRef(value.data(), value.length())
    ).getAsOpaquePointer();
}

const void* IntegerAttr_get(MLIRContext &context, rust::Str value)
{
    auto int_val = llvm::APSInt(llvm::StringRef(value.data(), value.length()));
    return IntegerAttr::get(&context, int_val).getAsOpaquePointer();
}

const void* BoolAttr_get(MLIRContext &context, bool value)
{
    return BoolAttr::get(&context, value).getAsOpaquePointer();
}

const void* DenseElementsAttr_get(
    const void* shaped_type,
    rust::Slice<const void *const> values
)
{
    std::vector<Attribute> values_vec;

    for (const auto &value : values)
        values_vec.push_back(Attribute::getFromOpaquePointer(value));

    return DenseElementsAttr::get(
        ShapedType::getFromOpaquePointer(shaped_type), values_vec
        )
        .getAsOpaquePointer();
}

const void* DenseFPElementsAttr_get(
    const void* shaped_type,
    rust::Slice<const void *const> values
)
{
    std::vector<Attribute> values_vec;

    for (const auto &value : values)
        values_vec.push_back(Attribute::getFromOpaquePointer(value));

    return DenseFPElementsAttr::get(
        ShapedType::getFromOpaquePointer(shaped_type), values_vec
        )
        .getAsOpaquePointer();
}

const void* DenseIntElementsAttr_get(
    const void* shaped_type,
    rust::Slice<const void *const> values
)
{
    std::vector<Attribute> values_vec;

    for (const auto &value : values)
        values_vec.push_back(Attribute::getFromOpaquePointer(value));

    return DenseIntElementsAttr::get(
        ShapedType::getFromOpaquePointer(shaped_type), values_vec
        )
        .getAsOpaquePointer();
}

const void* DictionaryAttr_get(
    MLIRContext &context,
    rust::Slice<const NamedAttribute *const> values
)
{
    std::vector<NamedAttribute> values_vec;

    for (const auto &value : values)
        values_vec.push_back(*value);

    return DictionaryAttr::get(&context, values_vec).getAsOpaquePointer();
}

} // namespace mithril_oxide_sys
