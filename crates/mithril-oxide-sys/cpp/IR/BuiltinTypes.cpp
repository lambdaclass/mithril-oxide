#include "BuiltinTypes.hpp"

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>

namespace mithril_oxide_sys {

std::unique_ptr<IntegerType> IntegerType_get(MLIRContext &context, unsigned int width, bool has_sign, bool is_signed)
{
    IntegerType::SignednessSemantics semantics;
    if(has_sign)
        semantics = is_signed
            ? IntegerType::SignednessSemantics::Signed
            : IntegerType::SignednessSemantics::Unsigned;
    else
        semantics = IntegerType::SignednessSemantics::Signless;

    return std::make_unique<IntegerType>(IntegerType::get(&context, width, semantics));
}

#define MITHRIL_CAST_TO_SHAPED_TYPE_IMPL(FROM_TYPE) std::unique_ptr<ShapedType> FROM_TYPE ## _to_ShapedType(const FROM_TYPE &x) \
    { \
         return std::make_unique<ShapedType>(x); \
    }

MITHRIL_CAST_TO_SHAPED_TYPE_IMPL(TensorType);
MITHRIL_CAST_TO_SHAPED_TYPE_IMPL(RankedTensorType);
MITHRIL_CAST_TO_SHAPED_TYPE_IMPL(VectorType);
MITHRIL_CAST_TO_SHAPED_TYPE_IMPL(MemRefType);

} // namespace mithril_oxide_sys
