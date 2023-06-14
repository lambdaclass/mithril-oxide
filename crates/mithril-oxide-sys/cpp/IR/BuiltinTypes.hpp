#pragma once

#include "BuiltinAttributes.hpp"

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>


namespace mithril_oxide_sys {

using mlir::MLIRContext;

using mlir::AffineExpr;
using mlir::BaseMemRefType;
using mlir::FloatType;
using mlir::FunctionType;
using mlir::IndexType;
using mlir::IntegerType;
using mlir::MemRefType;
using mlir::RankedTensorType;
using mlir::TensorType;
using mlir::VectorType;
using mlir::Type;

std::unique_ptr<IntegerType> IntegerType_get(MLIRContext &context, unsigned int width, bool has_sign, bool is_signed);

std::unique_ptr<FunctionType> FunctionType_get(
    MLIRContext &context,
    rust::Slice<const Type *const> inputs,
    rust::Slice<const Type *const> results);

// static FunctionType get(::mlir::MLIRContext *context, TypeRange inputs, TypeRange results);

#define MITHRIL_CAST_TO_SHAPED_TYPE(FROM_TYPE) std::unique_ptr<ShapedType> FROM_TYPE ## _to_ShapedType(const FROM_TYPE &x)

MITHRIL_CAST_TO_SHAPED_TYPE(TensorType);
MITHRIL_CAST_TO_SHAPED_TYPE(RankedTensorType);
MITHRIL_CAST_TO_SHAPED_TYPE(VectorType);
MITHRIL_CAST_TO_SHAPED_TYPE(MemRefType);

} // namespace mithril_oxide_sys
