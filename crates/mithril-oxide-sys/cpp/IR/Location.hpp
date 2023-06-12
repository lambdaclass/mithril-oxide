#pragma once

#include <memory>

#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys {

using mlir::CallSiteLoc;
using mlir::FileLineColLoc;
using mlir::FusedLoc;
using mlir::Location;
using mlir::MLIRContext;
using mlir::NameLoc;
using mlir::OpaqueLoc;
using mlir::UnknownLoc;


// TODO: Proxies for CallSiteLoc.
// TODO: Proxies for FileLineColLoc.
// TODO: Proxies for FusedLoc.
// TODO: Proxies for Location.
// TODO: Proxies for NameLoc.
// TODO: Proxies for OpaqueLoc.

std::unique_ptr<UnknownLoc> UnknownLoc_get(MLIRContext &ctx);
std::unique_ptr<Location> UnknownLoc_to_Location(const UnknownLoc &loc);

} // namespace mithril_oxide_sys
