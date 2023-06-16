#pragma once

#include <memory>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <rust/cxx.h>

#include "../../../lib.hpp"


namespace mithril_oxide_sys {

using mlir::DictionaryAttr;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::func::CallOp;
using mlir::FunctionType;
using mlir::Location;
using mlir::NamedAttribute;
using mlir::Value;
using mlir::Type;


std::unique_ptr<FuncOp> FuncOp_create(
    const Location &loc,
    rust::Str name,
    // FunctionType
    const void *function_type,
    rust::Slice<const NamedAttribute *const > attrs,
    // DictionaryAttr
    rust::Slice<const void *const > argAttrs
);

std::unique_ptr<ReturnOp> ReturnOp_create(
    const Location &loc,
    rust::Slice<const void *const > operands
);

std::unique_ptr<CallOp> CallOp_create(
    const Location &loc,
    rust::Slice<const void *const > results,
    rust::Slice<const void *const > operands
);

} // namespace mithril_oxide_sys
