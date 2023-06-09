#pragma once

#include <cstddef>

#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>


static void _OperationState_addOperands(
    mlir::OperationState &state,
    mlir::Value *data,
    std::size_t length
)
{
    mlir::ArrayRef<mlir::Value> slice(data, length);
    state.addOperands(slice);
}

static void _OperationState_addTypes(
    mlir::OperationState &state,
    mlir::Type *data,
    std::size_t length
)
{
    mlir::ArrayRef<mlir::Type> slice(data, length);
    state.addTypes(slice);
}
