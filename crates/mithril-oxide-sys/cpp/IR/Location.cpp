#include "Location.hpp"

#include <memory>

#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys {

// TODO: Proxies for CallSiteLoc.
// TODO: Proxies for FileLineColLoc.
// TODO: Proxies for FusedLoc.
// TODO: Proxies for Location.
// TODO: Proxies for NameLoc.
// TODO: Proxies for OpaqueLoc.

std::unique_ptr<UnknownLoc> UnknownLoc_get(MLIRContext &ctx)
{
    return std::make_unique<UnknownLoc>(UnknownLoc::get(&ctx));
}

std::unique_ptr<Location> UnknownLoc_to_Location(const UnknownLoc &loc)
{
    return std::make_unique<Location>(loc);
}

} // namespace mithril_oxide_sys
