#pragma once

#include <memory>

#include <mlir/IR/DialectRegistry.h>

#include "../lib.hpp"


namespace mithril_oxide_sys {

using mlir::DialectRegistry;


std::unique_ptr<DialectRegistry> DialectRegistry_new();

} // namespace mithril_oxide_sys
