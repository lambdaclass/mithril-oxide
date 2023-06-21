#include "Block.hpp"

#include <memory>

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

void Block_addArgument(Block &block, const void*type, const void* loc)
{
    block.addArgument(Type::getFromOpaquePointer(type), Location::getFromOpaquePointer(loc));
}

void* Block_getArgument(Block &block, unsigned i)
{
    return block.getArgument(i).getAsOpaquePointer();
}

rust::String Block_print(Block &block)
{
    std::string s;
    llvm::raw_string_ostream ss(s);
    block.print(ss);
    return rust::String::lossy(s);
}

} // namespace mithril_oxide_sys
