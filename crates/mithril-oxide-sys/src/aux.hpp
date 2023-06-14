#pragma once

#include <iostream>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>


mlir::ModuleOp ModuleOp_create(mlir::Location &loc)
{
    return mlir::ModuleOp::create(loc);
}

mlir::MLIRContext *UnknownLoc_getContext(mlir::UnknownLoc &loc)
{
    mlir::MLIRContext *ctx = loc.getContext();
    std::cout << "[C++] loc.getContext() = " << ctx << std::endl;
    return ctx;
}

mlir::Location UnknownLoc_to_Location(mlir::UnknownLoc &loc)
{
    mlir::Location generic_loc = loc;
    std::cout << "[C++] loc.getContext() = " << generic_loc->getContext() << std::endl;
    return generic_loc;
}

void test_ptr_cpp()
{
    mlir::MLIRContext ctx;

    mlir::UnknownLoc loc = mlir::UnknownLoc::get(&ctx);
    std::cout << "[Exclusively C++] ptr(context)     = " << ((void *) &ctx) << std::endl;
    std::cout << "[Exclusively C++] loc.getContext() = " << ((void *) loc.getContext()) << std::endl;
}
