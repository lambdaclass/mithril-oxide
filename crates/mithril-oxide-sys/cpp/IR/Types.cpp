#include "Types.hpp"

#include <memory>

namespace mithril_oxide_sys {

std::unique_ptr<Type> Value_getType(const Value& value)
{
    return std::make_unique<Type>(value.getType());
}

#define MITHRIL_CAST_TO_TYPE_IMPL(FROM_TYPE) std::unique_ptr<Type> FROM_TYPE ## _to_Type(const FROM_TYPE &x) \
    { \
         return std::make_unique<Type>(x); \
    }

MITHRIL_CAST_TO_TYPE_IMPL(BaseMemRefType);
MITHRIL_CAST_TO_TYPE_IMPL(FloatType);
MITHRIL_CAST_TO_TYPE_IMPL(FunctionType);
MITHRIL_CAST_TO_TYPE_IMPL(IndexType);
MITHRIL_CAST_TO_TYPE_IMPL(IntegerType);
MITHRIL_CAST_TO_TYPE_IMPL(MemRefType);
MITHRIL_CAST_TO_TYPE_IMPL(RankedTensorType);
MITHRIL_CAST_TO_TYPE_IMPL(TensorType);
MITHRIL_CAST_TO_TYPE_IMPL(VectorType);
MITHRIL_CAST_TO_TYPE_IMPL(ShapedType);

} // namespace mithril_oxide_sys
