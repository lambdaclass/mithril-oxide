#pragma once

#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <rust/cxx.h>

#include "../../lib.hpp"


namespace mithril_oxide_sys::llvm {

using mlir::LLVM::LLVMArrayType;
using mlir::LLVM::LLVMFixedVectorType;
using mlir::LLVM::LLVMFunctionType;
using mlir::LLVM::LLVMMetadataType;
using mlir::LLVM::LLVMPointerType;
using mlir::LLVM::LLVMPPCFP128Type;
using mlir::LLVM::LLVMScalableVectorType;
using mlir::LLVM::LLVMStructType;
using mlir::LLVM::LLVMTokenType;
using mlir::LLVM::LLVMVoidType;
using mlir::LLVM::LLVMX86MMXType;
using mlir::MLIRContext;


const void *LLVMArrayType_get(const void *elementType, unsigned int numElements);
const void *LLVMFixedVectorType_get(const void *elementType, unsigned int numElements);
const void *LLVMFunctionType_get(
    const void *result,
    rust::Slice<const void *const> arguments,
    bool isVarArg
);
const void *LLVMMetadataType_get(MLIRContext *ctx);
const void *LLVMPointerType_get(const void *elementType);
const void *LLVMPPCFP128Type_get(MLIRContext *ctx);
const void *LLVMScalableVectorType_get(const void *elementType, unsigned int numElements);
const void *LLVMStructType_getLiteral(
    MLIRContext *ctx,
    rust::Slice<const void *const> fields,
    bool isPacked
);
const void *LLVMTokenType_get(MLIRContext *ctx);
const void *LLVMVoidType_get(MLIRContext *ctx);
const void *LLVMX86MMXType_get(MLIRContext *ctx);

} // namespace mithril_oxide_sys::llvm
