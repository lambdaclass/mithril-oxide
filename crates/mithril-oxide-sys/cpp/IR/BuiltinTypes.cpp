#include "BuiltinTypes.hpp"

#include <memory>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/TypeRange.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

const void* IntegerType_get(MLIRContext &context, unsigned int width, bool has_sign, bool is_signed)
{
    IntegerType::SignednessSemantics semantics;
    if(has_sign)
        semantics = is_signed
            ? IntegerType::SignednessSemantics::Signed
            : IntegerType::SignednessSemantics::Unsigned;
    else
        semantics = IntegerType::SignednessSemantics::Signless;

    return IntegerType::get(&context, width, semantics).getAsOpaquePointer();
}

const void* FunctionType_get(
    MLIRContext &context,
    rust::Slice<const void *const> inputs,
    rust::Slice<const void *const> results)
{
    std::vector<Type> inputs_vec;
    std::vector<Type> results_vec;

    for (const auto &value : inputs)
        inputs_vec.push_back(Type::getFromOpaquePointer(value));
    for (const auto &value : results)
        results_vec.push_back(Type::getFromOpaquePointer(value));

    return FunctionType::get(&context, inputs_vec, results_vec).getAsOpaquePointer();
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
