#pragma once

#include <memory>

#include <mlir/IR/Value.h>
#include <rust/cxx.h>

#include "../lib.hpp"


namespace mithril_oxide_sys {

using mlir::Value;


rust::String Value_print(void *value);

void Value_dump(void *value);

} // namespace mithril_oxide_sys
