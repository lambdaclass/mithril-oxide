#include "LLVMTypes.hpp"

#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys::llvm {

const void *LLVMArrayType_get(const void *elementType, unsigned int numElements)
{
    return LLVMArrayType::get(
        mlir::Type::getFromOpaquePointer(elementType),
        numElements
    ).getAsOpaquePointer();
}

const void *LLVMFixedVectorType_get(const void *elementType, unsigned int numElements)
{
    return LLVMFixedVectorType::get(
        mlir::Type::getFromOpaquePointer(elementType),
        numElements
    ).getAsOpaquePointer();
}

const void *LLVMFunctionType_get(const void *result, rust::Slice<const void*> arguments, bool isVarArg)
{
    return LLVMFunctionType::get(
        mlir::Type::getFromOpaquePointer(result),
        mlir::ArrayRef(reinterpret_cast<mlir::Type *>(arguments.data()), arguments.length()),
        isVarArg
    ).getAsOpaquePointer();
}

const void *LLVMMetadataType_get(MLIRContext *ctx)
{
    return LLVMMetadataType::get(ctx).getAsOpaquePointer();
}

const void *LLVMPointerType_get(
    MLIRContext *ctx,
    const void *elementType,
    unsigned int addressSpace
)
{
    if (elementType == nullptr) {
        return LLVMPointerType::get(ctx, addressSpace).getAsOpaquePointer();
    } else {
        return LLVMPointerType::get(
            ctx,
            mlir::Type::getFromOpaquePointer(elementType),
            addressSpace
        ).getAsOpaquePointer();
    }
}

const void *LLVMPPCFP128Type_get(MLIRContext *ctx)
{
    return LLVMPPCFP128Type::get(ctx).getAsOpaquePointer();
}

const void *LLVMScalableVectorType_get(const void *elementType, unsigned int numElements)
{
    return LLVMScalableVectorType::get(
        mlir::Type::getFromOpaquePointer(elementType),
        numElements
    ).getAsOpaquePointer();
}

const void *LLVMStructType_getLiteral(
    MLIRContext *ctx,
    rust::Slice<const void*> fields,
    bool isPacked
)
{
    return LLVMStructType::getLiteral(
        ctx,
        mlir::ArrayRef(reinterpret_cast<mlir::Type *>(fields.data()), fields.length()),
        isPacked
    ).getAsOpaquePointer();
}

const void *LLVMTokenType_get(MLIRContext *ctx)
{
    return LLVMTokenType::get(ctx).getAsOpaquePointer();
}

const void *LLVMVoidType_get(MLIRContext *ctx)
{
    return LLVMVoidType::get(ctx).getAsOpaquePointer();
}

const void *LLVMX86MMXType_get(MLIRContext *ctx)
{
    return LLVMX86MMXType::get(ctx).getAsOpaquePointer();
}

} // namespace mithril_oxide_sys::llvm
