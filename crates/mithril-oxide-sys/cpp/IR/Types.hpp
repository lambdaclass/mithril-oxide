#pragma once

#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>


namespace mithril_oxide_sys {

using mlir::Type;
using mlir::Value;

using mlir::BaseMemRefType;
using mlir::FloatType;
using mlir::FunctionType;
using mlir::IndexType;
using mlir::IntegerType;
using mlir::MemRefType;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::TensorType;
using mlir::VectorType;

// using mlir::AffineExpr;


std::unique_ptr<Type> Value_getType(const Value& value);

#define MITHRIL_CAST_TO_TYPE(FROM_TYPE) std::unique_ptr<Type> FROM_TYPE ## _to_Type(const FROM_TYPE &x)

MITHRIL_CAST_TO_TYPE(BaseMemRefType);
MITHRIL_CAST_TO_TYPE(FloatType);
MITHRIL_CAST_TO_TYPE(FunctionType);
MITHRIL_CAST_TO_TYPE(IndexType);
MITHRIL_CAST_TO_TYPE(IntegerType);
MITHRIL_CAST_TO_TYPE(MemRefType);
MITHRIL_CAST_TO_TYPE(RankedTensorType);
MITHRIL_CAST_TO_TYPE(TensorType);
MITHRIL_CAST_TO_TYPE(VectorType);
MITHRIL_CAST_TO_TYPE(ShapedType);

} // namespace mithril_oxide_sys
