#include "BuiltinTypes.hpp"

#include <memory>
#include <mlir/IR/BuiltinAttributes.h>

namespace mithril_oxide_sys
{

std::unique_ptr<IntegerType> IntegerType_get(MLIRContext &context, unsigned int width, bool has_sign, bool is_signed)
{
    IntegerType::SignednessSemantics semantics;
    if(has_sign)
    {
        semantics = is_signed ? IntegerType::SignednessSemantics::Signed : IntegerType::SignednessSemantics::Unsigned;
    }
    else
    {
        semantics = IntegerType::SignednessSemantics::Signless;
    }

    return std::make_unique<IntegerType>(IntegerType::get(&context, width, semantics));
}

} // namespace mithril_oxide_sys
