#pragma once

#include <memory>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <rust/cxx.h>


namespace mithril_oxide_sys {

using mlir::NamedAttribute;
using mlir::Attribute;
using mlir::StringAttr;

rust::String Attribute_print(const Attribute &op);

std::unique_ptr<NamedAttribute> NamedAttribute_new(const StringAttr &name, const Attribute &attr);
rust::Str NamedAttribute_getName(const NamedAttribute &attr);
std::unique_ptr<Attribute> NamedAttribute_getValue(const NamedAttribute &attr);

} // namespace mithril_oxide_sys
