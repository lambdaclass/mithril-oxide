#include "BuiltinAttributes.hpp"

#include <memory>

#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

const void* UnitAttr_get(MLIRContext &context)
{
    return UnitAttr::get(&context).getAsOpaquePointer();
}

const void* TypeAttr_get(const void* type)
{
    return TypeAttr::get(Type::getFromOpaquePointer(type)).getAsOpaquePointer();
}

const void* StridedLayoutAttr_get(
    MLIRContext &context,
    rust::i64 offset,
    rust::Slice<const rust::i64> strides)
{
    return StridedLayoutAttr::get(&context, offset, mlir::ArrayRef(strides.data(), strides.length())).getAsOpaquePointer();
}

const void* OpaqueAttr_get(
    const void* dialect, // string_attr
    rust::Str attr_data,
    const void* type)
{
    return OpaqueAttr::get(
            StringAttr::getFromOpaquePointer(dialect),
            mlir::StringRef(attr_data.data(), attr_data.length()),
            Type::getFromOpaquePointer(type))
        .getAsOpaquePointer();
}

const void* SparseElementsAttr_get(
    const void* shaped_type, // ShapedType
    const void* indices, // DenseElementsAttr
    const void* values // DenseElementsAttr
    )
{
    return SparseElementsAttr::get(
            ShapedType::getFromOpaquePointer(shaped_type),
            DenseElementsAttr::getFromOpaquePointer(indices).cast<DenseElementsAttr>(),
            DenseElementsAttr::getFromOpaquePointer(values).cast<DenseElementsAttr>()
        )
        .getAsOpaquePointer();
}

const void* StringAttr_get(MLIRContext &context, rust::Str value)
{
    return StringAttr::get(
        &context,
        mlir::StringRef(value.data(), value.length())
    ).getAsOpaquePointer();
}

const void* FlatSymbolRefAttr_get(MLIRContext &context, rust::Str value)
{
    return FlatSymbolRefAttr::get(
        &context,
        mlir::StringRef(value.data(), value.length())
    ).getAsOpaquePointer();
}

const void* IntegerAttr_get(MLIRContext &context, rust::Str value)
{
    auto int_val = llvm::APSInt(llvm::StringRef(value.data(), value.length()));
    return IntegerAttr::get(&context, int_val).getAsOpaquePointer();
}

const void* BoolAttr_get(MLIRContext &context, bool value)
{
    return BoolAttr::get(&context, value).getAsOpaquePointer();
}

const void* DenseElementsAttr_get(
    const void* shaped_type,
    rust::Slice<const void *const> values
)
{
    std::vector<Attribute> values_vec;

    for (const auto &value : values)
        values_vec.push_back(Attribute::getFromOpaquePointer(value));

    return DenseElementsAttr::get(
        ShapedType::getFromOpaquePointer(shaped_type), values_vec
        )
        .getAsOpaquePointer();
}

const void* DenseFPElementsAttr_get(
    const void* shaped_type,
    rust::Slice<const void *const> values
)
{
    std::vector<Attribute> values_vec;

    for (const auto &value : values)
        values_vec.push_back(Attribute::getFromOpaquePointer(value));

    return DenseFPElementsAttr::get(
        ShapedType::getFromOpaquePointer(shaped_type), values_vec
        )
        .getAsOpaquePointer();
}

const void* DenseIntElementsAttr_get(
    const void* shaped_type,
    rust::Slice<const void *const> values
)
{
    std::vector<Attribute> values_vec;

    for (const auto &value : values)
        values_vec.push_back(Attribute::getFromOpaquePointer(value));

    return DenseIntElementsAttr::get(
        ShapedType::getFromOpaquePointer(shaped_type), values_vec
        )
        .getAsOpaquePointer();
}

const void* DenseI64ElementsAttr_get(
    const void* shaped_type,
    rust::Slice<const void *const> values
)
{
    std::vector<Attribute> values_vec;

    for (const auto &value : values)
        values_vec.push_back(Attribute::getFromOpaquePointer(value));

    return DenseIntElementsAttr::get(
        ShapedType::getFromOpaquePointer(shaped_type), values_vec
        )
        .getAsOpaquePointer();
}

const void* DictionaryAttr_get(
    MLIRContext &context,
    rust::Slice<const NamedAttribute *const> values
)
{
    std::vector<NamedAttribute> values_vec;

    for (const auto &value : values)
        values_vec.push_back(*value);

    return DictionaryAttr::get(&context, values_vec).getAsOpaquePointer();
}

const void* ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<const void* const > values
)
{
    std::vector<Attribute> values_vec;

    for (const auto &value : values)
        values_vec.push_back(Attribute::getFromOpaquePointer(value));

    return ArrayAttr::get(&context, values_vec).getAsOpaquePointer();
}

const void* DenseBoolArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<bool> values
)
{
    return DenseBoolArrayAttr::get(&context, mlir::ArrayRef(values.data(), values.size())).getAsOpaquePointer();
}

const void* DenseI8ArrayAttr_get(
    MLIRContext &context,
    rust::Slice<rust::i8> values
)
{
    return DenseI8ArrayAttr::get(&context, mlir::ArrayRef(values.data(), values.size())).getAsOpaquePointer();
}

const void* DenseI16ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<rust::i16> values
)
{
    return DenseI16ArrayAttr::get(&context, mlir::ArrayRef(values.data(), values.size())).getAsOpaquePointer();
}

const void* DenseI32ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<rust::i32> values
)
{
    return DenseI32ArrayAttr::get(&context, mlir::ArrayRef(values.data(), values.size())).getAsOpaquePointer();
}

const void* DenseI64ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<rust::i64> values
)
{
    return DenseI64ArrayAttr::get(&context, mlir::ArrayRef(values.data(), values.size())).getAsOpaquePointer();
}

const void* DenseF32ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<rust::f32> values
)
{
    return DenseF32ArrayAttr::get(&context, mlir::ArrayRef(values.data(), values.size())).getAsOpaquePointer();
}

const void* DenseF64ArrayAttr_get(
    MLIRContext &context,
    const rust::Slice<rust::f64> values
)
{
    return DenseF64ArrayAttr::get(&context, mlir::ArrayRef(values.data(), values.size())).getAsOpaquePointer();
}

} // namespace mithril_oxide_sys
