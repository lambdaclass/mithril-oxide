#pragma once

#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <rust/cxx.h>

#include "../lib.hpp"


namespace mithril_oxide_sys {

using mlir::Type;
using mlir::Value;

// using mlir::AffineExpr;


const void* Value_getType(const void* value);

rust::String Type_print(const void* type);
unsigned int Type_getIntOrFloatBitWidth(const void* type);
void Type_dump(const void* type);

} // namespace mithril_oxide_sys
