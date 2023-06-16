#include "EmitC.hpp"

#include <mlir/Dialect/EmitC/IR/EmitC.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys::emitc {

const void *OpaqueType_get(MLIRContext *ctx, rust::Str name)
{
    return OpaqueType::get(ctx, mlir::StringRef(name.data(), name.length())).getAsOpaquePointer();
}

const void *PointerType_get(const void *type)
{
    return PointerType::get(mlir::Type::getFromOpaquePointer(type)).getAsOpaquePointer();
}

} // mithril_oxide_sys::emitc
