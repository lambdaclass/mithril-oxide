#include "Operation.hpp"

#include <mlir/IR/Operation.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

rust::Str Operation_getName(Operation &op)
{
    auto ref = op.getName().getStringRef();
    return rust::Str(ref.data(), ref.size());
}

rust::String Operation_print(Operation &op)
{
    std::string s;
    llvm::raw_string_ostream ss(s);
    op.print(ss);
    return rust::String::lossy(s);
}

} // namespace mithril_oxide_sys
