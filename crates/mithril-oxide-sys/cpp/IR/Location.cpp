#include "Location.hpp"

#include <memory>

#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>


namespace mithril_oxide_sys {

// TODO: Proxies for FusedLoc.
// TODO: Proxies for Location.
// TODO: Proxies for NameLoc.
// TODO: Proxies for OpaqueLoc.

const void* UnknownLoc_get(MLIRContext &ctx)
{
    return UnknownLoc::get(&ctx).getAsOpaquePointer();
}

const void* FileLineColLoc_get(const void* filename_ptr, unsigned line, unsigned column)
{
    auto filename = mlir::StringAttr::getFromOpaquePointer(filename_ptr);
    return FileLineColLoc::get(filename, line, column).getAsOpaquePointer();
}

const void* CallSiteLoc_get(const void* callee_ptr, const void* caller_ptr)
{
    return CallSiteLoc::get(
        Location::getFromOpaquePointer(callee_ptr),
        Location::getFromOpaquePointer(caller_ptr)
    ).getAsOpaquePointer();
}

} // namespace mithril_oxide_sys
