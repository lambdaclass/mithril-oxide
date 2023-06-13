#include "Value.hpp"

#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace mithril_oxide_sys
{

std::unique_ptr<Value> BlockArgument_toValue(const BlockArgument& arg)
{
    return std::make_unique<Value>(arg);
}

rust::String Value_print(Value &value)
{
    std::string s;
    llvm::raw_string_ostream ss(s);
    value.print(ss);
    return rust::String::lossy(s);
}

} // namespace mithril_oxide_sys
