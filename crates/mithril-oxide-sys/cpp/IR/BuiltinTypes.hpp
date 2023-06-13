#pragma once

#include "BuiltinAttributes.hpp"
#include <memory>
#include <mlir/IR/BuiltinTypes.h>


namespace mithril_oxide_sys {

using mlir::MLIRContext;

using mlir::FunctionType;
using mlir::IntegerType;
using mlir::FloatType;
using mlir::TensorType;
using mlir::BaseMemRefType;
using mlir::MemRefType;
using mlir::RankedTensorType;
using mlir::VectorType;
using mlir::AffineExpr;
using mlir::IndexType;

std::unique_ptr<IntegerType> IntegerType_get(MLIRContext &context, unsigned int width, bool has_sign, bool is_signed);

//  static IntegerType get(::mlir::MLIRContext *context, unsigned width, SignednessSemantics signedness = Signless);

} // namespace mithril_oxide_sys
