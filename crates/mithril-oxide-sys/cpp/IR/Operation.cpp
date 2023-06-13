#include "Operation.hpp"

#include <mlir/IR/Operation.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

rust::Str Operation_getName(Operation &op)
{
    auto ref = op.getName().getStringRef();
    return rust::Str(ref.data(), ref.size());
}

} // namespace mithril_oxide_sys
