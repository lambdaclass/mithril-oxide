#pragma once

#include <memory>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

using mlir::Attribute;
using mlir::BoolAttr;
using mlir::DenseElementsAttr;
using mlir::DenseFPElementsAttr;
using mlir::DenseIntElementsAttr;
using mlir::DictionaryAttr;
using mlir::FlatSymbolRefAttr;
using mlir::FloatAttr;
using mlir::IntegerAttr;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::ShapedType;
using mlir::StringAttr;


std::unique_ptr<StringAttr> StringAttr_get(MLIRContext &context, rust::Str value);
std::unique_ptr<IntegerAttr> IntegerAttr_get(MLIRContext &context, rust::Str value);
std::unique_ptr<BoolAttr> BoolAttr_get(MLIRContext &context, bool value);

std::unique_ptr<DenseElementsAttr> DenseElementsAttr_get(
    const ShapedType &type,
    rust::Slice<const Attribute *const> values
);

std::unique_ptr<DictionaryAttr> DictionaryAttr_get(
    MLIRContext &context,
    rust::Slice<const NamedAttribute *const> values
);

// static DictionaryAttr get(::mlir::MLIRContext *context, ArrayRef<NamedAttribute> value = std::nullopt);

#define MITHRIL_CAST_TO_ATTR(FROM_TYPE) std::unique_ptr<Attribute> FROM_TYPE ## _to_Attribute(const FROM_TYPE &x)

MITHRIL_CAST_TO_ATTR(DictionaryAttr);
MITHRIL_CAST_TO_ATTR(StringAttr);
MITHRIL_CAST_TO_ATTR(FloatAttr);
MITHRIL_CAST_TO_ATTR(IntegerAttr);
MITHRIL_CAST_TO_ATTR(DenseElementsAttr);
MITHRIL_CAST_TO_ATTR(DenseIntElementsAttr);
MITHRIL_CAST_TO_ATTR(DenseFPElementsAttr);
MITHRIL_CAST_TO_ATTR(BoolAttr);
MITHRIL_CAST_TO_ATTR(FlatSymbolRefAttr);

} // namespace mithril_oxide_sys
