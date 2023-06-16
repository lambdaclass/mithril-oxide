#include "MLProgramTypes.hpp"

#include <mlir/Dialect/MLProgram/IR/MLProgramTypes.h>
#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys::ml_program {

const void *TokenType_get(MLIRContext *ctx)
{
    return TokenType::get(ctx).getAsOpaquePointer();
}

} // mithril_oxide_sys::ml_program
