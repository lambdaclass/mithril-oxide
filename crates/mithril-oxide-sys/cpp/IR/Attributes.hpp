#pragma once

#include <mlir/IR/Attributes.h>

#include <rust/cxx.h>


namespace mithril_oxide_sys {

using mlir::NamedAttribute;
using mlir::Attribute;

rust::String Attribute_print(const Attribute &op);

} // namespace mithril_oxide_sys
