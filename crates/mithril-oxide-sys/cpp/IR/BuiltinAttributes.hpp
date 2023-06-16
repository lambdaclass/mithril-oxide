#pragma once

#include <memory>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <rust/cxx.h>

#include "../lib.hpp"


namespace mithril_oxide_sys {

using mlir::ArrayAttr;
using mlir::Attribute;
using mlir::BoolAttr;
using mlir::DenseBoolArrayAttr;
using mlir::DenseElementsAttr;
using mlir::DenseF32ArrayAttr;
using mlir::DenseF64ArrayAttr;
using mlir::DenseFPElementsAttr;
using mlir::DenseI16ArrayAttr;
using mlir::DenseI32ArrayAttr;
using mlir::DenseI64ArrayAttr;
using mlir::DenseI8ArrayAttr;
using mlir::DenseIntElementsAttr;
using mlir::DictionaryAttr;
using mlir::FlatSymbolRefAttr;
using mlir::FloatAttr;
using mlir::IntegerAttr;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::OpaqueAttr;
using mlir::ShapedType;
using mlir::SparseElementsAttr;
using mlir::StridedLayoutAttr;
using mlir::StringAttr;
using mlir::Type;
using mlir::TypeAttr;
using mlir::UnitAttr;


const void* StringAttr_get(MLIRContext &context, rust::Str value);
const void* TypeAttr_get(const void* type);
const void* UnitAttr_get(MLIRContext &context);
const void* FlatSymbolRefAttr_get(MLIRContext &context, rust::Str value);
const void* IntegerAttr_get(MLIRContext &context, rust::Str value);
const void* BoolAttr_get(MLIRContext &context, bool value);

const void* StridedLayoutAttr_get(
    MLIRContext &context,
    rust::i64 offset,
    rust::Slice<const rust::i64> strides);

const void* OpaqueAttr_get(
    const void* dialect, // string_attr
    rust::Str attr_data,
    const void* type);

const void* SparseElementsAttr_get(
    const void* shaped_type, // ShapedType
    const void* indices, // DenseElementsAttr
    const void* values // DenseElementsAttr
    );

const void* DenseElementsAttr_get(
    const void* shaped_type, // ShapedType trait
    // Attribute
    rust::Slice<const void *const> values
);

const void* DenseFPElementsAttr_get(
    const void* shaped_type, // ShapedType trait
    // Attribute
    rust::Slice<const void *const> values
);

const void* DenseIntElementsAttr_get(
    const void* shaped_type, // ShapedType trait
    // Attribute
    rust::Slice<const void *const> values
);

const void* DictionaryAttr_get(
    MLIRContext &context,
    rust::Slice<const NamedAttribute *const> values
);

const void* ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<const void* const > values
);

const void* DenseBoolArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<const bool> values
);

const void* DenseI8ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<const rust::i8> values
);

const void* DenseI16ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<const rust::i16> values
);

const void* DenseI32ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<const rust::i32> values
);

const void* DenseI64ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<const rust::i64> values
);

const void* DenseF32ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<const rust::f32> values
);

const void* DenseF64ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<const rust::f64> values
);

} // namespace mithril_oxide_sys
