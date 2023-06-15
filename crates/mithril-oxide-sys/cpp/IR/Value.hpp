#pragma once

#include <memory>

#include <mlir/IR/Value.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {


using mlir::Value;

using c_void = void;

rust::String Value_print(void *value);

void Value_dump(void *value);

} // namespace mithril_oxide_sys
