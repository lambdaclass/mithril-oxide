#include "OpenACC.hpp"

#include <mlir/Dialect/OpenACC/OpenACC.h>
#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys::acc {

void *DataBoundsType_create(MLIRContext *ctx)
{
    // FIXME: Use the real type (if available).
    return mlir::acc::GANG
}

} // namespace mithril_oxide_sys::acc
