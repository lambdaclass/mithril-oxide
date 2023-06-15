#include "Types.hpp"

#include <memory>

#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>


namespace mithril_oxide_sys {

const void* Value_getType(const void* value)
{
    auto val = mlir::Value::getFromOpaquePointer(value);
    return val.getType().getAsOpaquePointer();
}

unsigned int Type_getIntOrFloatBitWidth(const void* type)
{
    auto t = Type::getFromOpaquePointer(type);
    return t.getIntOrFloatBitWidth();
}

void Type_dump(const void* type)
{
     auto t = Type::getFromOpaquePointer(type);
    t.dump();
}

} // namespace mithril_oxide_sys
