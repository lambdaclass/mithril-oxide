#include "Operation.hpp"
#include "rust/cxx.h"

namespace mithril_oxide_sys {

rust::Str Operation_getName(Operation &op)
{
    auto ref = op.getName().getStringRef();
    return rust::Str(ref.data(), ref.size());
}

}
