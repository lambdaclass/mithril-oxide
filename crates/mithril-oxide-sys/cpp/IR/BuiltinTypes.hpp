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

std::unique_ptr<IntegerType> IntegerType_get(MLIRContext &context, unsigned int width, bool has_sign, bool is_signed);

} // namespace mithril_oxide_sys
