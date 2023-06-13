#include "FuncOps.hpp"

#include <memory>
#include <vector>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

std::unique_ptr<FuncOp> FuncOp_create(
    const Location &loc,
    rust::Str name,
    const FunctionType &type,
    rust::Slice<const NamedAttribute *const> attrs,
    rust::Slice<const DictionaryAttr *const> argAttrs
)
{
    std::vector<NamedAttribute> attrs_vec;
    std::vector<DictionaryAttr> argAttrs_vec;

    for (const auto &attr : attrs)
        attrs_vec.push_back(*attr);
    for (const auto &argAttr : argAttrs)
        argAttrs_vec.push_back(*argAttr);

    return std::make_unique<FuncOp>(FuncOp::create(
        loc,
        mlir::StringRef(name.data(), name.length()),
        type,
        mlir::ArrayRef(attrs_vec.data(), attrs_vec.size()),
        mlir::ArrayRef(argAttrs_vec.data(), argAttrs_vec.size())
    ));
}

} // namespace mithril_oxide_sys
