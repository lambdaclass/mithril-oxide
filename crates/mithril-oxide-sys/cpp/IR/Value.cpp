#include "Value.hpp"

#include <memory>
#include <string>

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Value.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

rust::String Value_print(void *value)
{
    auto val = Value::getFromOpaquePointer(value);
    std::string s;
    llvm::raw_string_ostream ss(s);
    val.print(ss);
    return rust::String::lossy(s);
}

void Value_dump(void *value)
{
    auto val = Value::getFromOpaquePointer(value);
    val.dump();
}

} // namespace mithril_oxide_sys
