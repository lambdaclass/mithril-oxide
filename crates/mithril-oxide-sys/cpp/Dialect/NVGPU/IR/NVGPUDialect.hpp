#pragma once

#include <mlir/Dialect/NVGPU/IR/NVGPUDialect.h>
#include <mlir/IR/MLIRContext.h>

#include "../../../lib.hpp"


namespace mithril_oxide_sys::nvgpu {

using mlir::MLIRContext;
using mlir::nvgpu::DeviceAsyncTokenType;


const void *DeviceAsyncTokenType_get(MLIRContext *ctx);

} // namespace mithril_oxide_sys::nvgpu
