#include "Operation.hpp"

#include <memory>
#include <string>

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
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

std::unique_ptr<OpResult> Operation_getResult(Operation &op, unsigned idx)
{
    return std::make_unique<OpResult>(op.getResult(idx));
}

} // namespace mithril_oxide_sys
