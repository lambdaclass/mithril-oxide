#pragma once

#include <memory>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <rust/cxx.h>

#include "../lib.hpp"


namespace mithril_oxide_sys {

using mlir::NamedAttribute;
using mlir::Attribute;
using mlir::StringAttr;


rust::String Attribute_print(const void* attr);

std::unique_ptr<NamedAttribute> NamedAttribute_new(const void* name, const void* attr);
rust::Str NamedAttribute_getName(const NamedAttribute &attr);
const void* NamedAttribute_getValue(const NamedAttribute &attr);

} // namespace mithril_oxide_sys
