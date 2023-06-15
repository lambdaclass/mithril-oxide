#pragma once

#include <memory>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

using c_void = void;

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


const void* StringAttr_get(MLIRContext &context, rust::Str value);
const void* IntegerAttr_get(MLIRContext &context, rust::Str value);
const void* BoolAttr_get(MLIRContext &context, bool value);

const void* DenseElementsAttr_get(
    const void* shaped_type, // ShapedType trait
    // Attribute
    rust::Slice<const void *const> values
);

const void* DictionaryAttr_get(
    MLIRContext &context,
    rust::Slice<const NamedAttribute *const> values
);

} // namespace mithril_oxide_sys
