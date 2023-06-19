#pragma once

#include <memory>

#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Types.h>

#include "../lib.hpp"


namespace mithril_oxide_sys {

using mlir::DataLayout;
using mlir::ModuleOp;
using mlir::Type;


std::unique_ptr<DataLayout> DataLayout_new(const ModuleOp &op);


unsigned DataLayout_getTypeSize(const DataLayout &layout, const void* type_ptr);
unsigned DataLayout_getTypeABIAlignment(const DataLayout &layout, const void* type_ptr);
unsigned DataLayout_getTypePreferredAlignment(const DataLayout &layout, const void* type_ptr);

} // namespace mithril_oxide_sys
