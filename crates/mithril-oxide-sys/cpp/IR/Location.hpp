#pragma once

#include <memory>

#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>

#include "../lib.hpp"


namespace mithril_oxide_sys {

using mlir::CallSiteLoc;
using mlir::FileLineColLoc;
using mlir::FusedLoc;
using mlir::Location;
using mlir::MLIRContext;
using mlir::NameLoc;
using mlir::OpaqueLoc;
using mlir::UnknownLoc;

// TODO: Proxies for FusedLoc.
// TODO: Proxies for Location.
// TODO: Proxies for NameLoc.
// TODO: Proxies for OpaqueLoc.

const void* UnknownLoc_get(MLIRContext &ctx);

// filename - stringattr
const void* FileLineColLoc_get(const void* filename, unsigned line, unsigned column);

// callee - location
// caller - location
const void* CallSiteLoc_get(const void* callee, const void* caller);

} // namespace mithril_oxide_sys
