#pragma once

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <rust/cxx.h>

#include "../lib.hpp"


namespace mithril_oxide_sys {

using mlir::AffineExpr;
using mlir::BaseMemRefType;
using mlir::FloatType;
using mlir::FunctionType;
using mlir::IndexType;
using mlir::IntegerType;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::TensorType;
using mlir::Type;
using mlir::VectorType;


const void* IntegerType_get(MLIRContext &context, unsigned int width, bool has_sign, bool is_signed);

const void* FunctionType_get(
    MLIRContext &context,
    rust::Slice<const void *const> inputs,
    rust::Slice<const void *const> results);

} // namespace mithril_oxide_sys
