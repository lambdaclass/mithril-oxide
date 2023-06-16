#include "NVGPUDialect.hpp"

#include <mlir/Dialect/NVGPU/IR/NVGPUDialect.h>
#include <mlir/IR/MLIRContext.h>


namespace mithril_oxide_sys::nvgpu {

const void *DeviceAsyncTokenType_get(MLIRContext *ctx)
{
    return DeviceAsyncTokenType::get(ctx).getAsOpaquePointer();
}

} // namespace mithril_oxide_sys::nvgpu
