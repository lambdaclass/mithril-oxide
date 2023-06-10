#pragma once

#include <cstddef>
#include <iostream>
#include <optional>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>


#define impl_downgrade(target_t, source_t, name) \
    target_t name(source_t value) \
    { \
        return value; \
    }


// Class downgrades (either by assignment or hierarchic casting).
impl_downgrade(mlir::Location, mlir::UnknownLoc, _UnknownLoc_downgradeTo_Location)
impl_downgrade(mlir::Operation *, mlir::ModuleOp &, _ModuleOp_downgradeTo_Operation)


void test_printLoadedDialects(mlir::MLIRContext *ctx)
{
    std::cout << "Registered dialects:" << std::endl;
    for (auto d : ctx->getAvailableDialects())
        std::cout << d.str() << std::endl;

    std::cout << "Loaded dialects:" << std::endl;
    for (const auto &d : ctx->getLoadedDialects())
        std::cout << d->getNamespace().str() << std::endl;
}


mlir::ModuleOp _ModuleOp_create(mlir::Location loc, const mlir::StringRef *name)
{
    test_printLoadedDialects(loc.getContext());
    std::optional<mlir::StringRef> name_ref = std::nullopt;
    if (name != nullptr)
        name_ref = *name;

    return mlir::ModuleOp::create(loc, name_ref);
}

void _OperationState_addOperands(
    mlir::OperationState &state,
    mlir::Value *data,
    std::size_t length
)
{
    mlir::ArrayRef<mlir::Value> slice(data, length);
    state.addOperands(slice);
}

void _OperationState_addTypes(
    mlir::OperationState &state,
    mlir::Type *data,
    std::size_t length
)
{
    mlir::ArrayRef<mlir::Type> slice(data, length);
    state.addTypes(slice);
}
