#include "Attributes.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>

namespace mithril_oxide_sys
{

rust::String Attribute_print(const Attribute &op)
{
    std::string s;
    llvm::raw_string_ostream ss(s);
    op.print(ss);
    return rust::String::lossy(s);
}

}
