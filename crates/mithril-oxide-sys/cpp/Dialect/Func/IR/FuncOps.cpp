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

std::unique_ptr<ReturnOp> ReturnOp_create(
    const Location &loc,
    rust::Slice<const Value *const > operands
)
{
    auto builder = mlir::OpBuilder(loc->getContext());
    auto state = mlir::OperationState(loc, ReturnOp::getOperationName());
    std::vector<Value> operands_vec;

    for (const auto &val : operands)
        operands_vec.push_back(*val);

    ReturnOp::build(builder, state, operands_vec);

    ReturnOp op = ReturnOp(builder.create(state));
    return std::make_unique<ReturnOp>(op);
}

std::unique_ptr<CallOp> CallOp_create(
    const Location &loc,
    rust::Slice<const Type *const > results,
    rust::Slice<const Value *const > operands
)
{
    auto builder = mlir::OpBuilder(loc->getContext());
    auto state = mlir::OperationState(loc, CallOp::getOperationName());
    std::vector<Value> operands_vec;
    std::vector<Type> results_vec;

    for (const auto &val : operands)
        operands_vec.push_back(*val);
    for (const auto &val : results)
        results_vec.push_back(*val);

    CallOp::build(builder, state, results_vec, operands_vec);

    auto op = CallOp(builder.create(state));
    return std::make_unique<CallOp>(op);
}

} // namespace mithril_oxide_sys
