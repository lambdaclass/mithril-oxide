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

rust::String Type_print(const void* type_ptr)
{
    auto type = Type::getFromOpaquePointer(type_ptr);
    std::string s;
    llvm::raw_string_ostream ss(s);
    type.print(ss);
    return rust::String::lossy(s);
}

} // namespace mithril_oxide_sys
