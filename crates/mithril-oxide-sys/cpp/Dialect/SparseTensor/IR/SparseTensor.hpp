#pragma once

#include <mlir/Dialect/SparseTensor/IR/SparseTensor.h>

#include "../../../lib.hpp"


namespace mithril_oxide_sys::sparse_tensor {

using mlir::sparse_tensor::StorageSpecifierType;


const void *StorageSpecifierType_get(const void *encoding);

} // namespace mithril_oxide_sys::sparse_tensor
