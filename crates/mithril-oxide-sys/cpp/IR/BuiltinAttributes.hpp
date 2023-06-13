#pragma once

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

using mlir::DictionaryAttr;
using mlir::MLIRContext;
using mlir::StringAttr;


std::unique_ptr<StringAttr> StringAttr_get(MLIRContext &context, rust::Str value);

} // namespace mithril_oxide_sys
