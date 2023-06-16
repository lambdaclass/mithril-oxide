#pragma once

#include <mlir/Dialect/MLProgram/IR/MLProgramTypes.h>
#include <mlir/IR/MLIRContext.h>

#include "../../../lib.hpp"


namespace mithril_oxide_sys::ml_program {

using mlir::ml_program::TokenType;
using mlir::MLIRContext;


const void *TokenType_get(MLIRContext *ctx);

} // mithril_oxide_sys::ml_program
