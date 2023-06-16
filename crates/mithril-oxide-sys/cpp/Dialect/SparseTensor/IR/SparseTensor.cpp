#include "SparseTensor.hpp"

#include <mlir/Dialect/SparseTensor/IR/SparseTensor.h>


namespace mithril_oxide_sys::sparse_tensor {

const void *StorageSpecifierType_get(const void *encoding)
{
    return StorageSpecifierType::get(
        mlir::sparse_tensor::SparseTensorEncodingAttr::getFromOpaquePointer(encoding)
    ).getAsOpaquePointer();
}

} // namespace mithril_oxide_sys::sparse_tensor
