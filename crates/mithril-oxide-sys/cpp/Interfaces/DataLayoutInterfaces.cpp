#include "DataLayoutInterfaces.hpp"

#include <memory>

#include <mlir/Interfaces/DataLayoutInterfaces.h>


namespace mithril_oxide_sys {

std::unique_ptr<DataLayout> DataLayout_new(const ModuleOp &op)
{
    return std::make_unique<DataLayout>(op);
}

unsigned DataLayout_getTypeSize(const DataLayout &layout, const void* type_ptr)
{
    auto type = Type::getFromOpaquePointer(type_ptr);
    return layout.getTypeSize(type);
}

unsigned DataLayout_getTypeABIAlignment(const DataLayout &layout, const void* type_ptr)
{
    auto type = Type::getFromOpaquePointer(type_ptr);
    return layout.getTypeABIAlignment(type);
}

unsigned DataLayout_getTypePreferredAlignment(const DataLayout &layout, const void* type_ptr)
{
    auto type = Type::getFromOpaquePointer(type_ptr);
    return layout.getTypePreferredAlignment(type);
}

} // namespace mithril_oxide_sys
