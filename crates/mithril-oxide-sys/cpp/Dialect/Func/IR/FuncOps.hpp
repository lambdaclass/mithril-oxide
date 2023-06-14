#pragma once

#include <memory>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

using mlir::DictionaryAttr;
using mlir::func::FuncOp;
using mlir::FunctionType;
using mlir::Location;
using mlir::NamedAttribute;


std::unique_ptr<FuncOp> FuncOp_create(
    const Location &loc,
    rust::Str name,
    const FunctionType &type,
    rust::Slice<const NamedAttribute *const > attrs,
    rust::Slice<const DictionaryAttr *const > argAttrs
);

} // namespace mithril_oxide_sys
