#include "Attributes.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>

#include <memory>

namespace mithril_oxide_sys
{

rust::String Attribute_print(const Attribute &op)
{
    std::string s;
    llvm::raw_string_ostream ss(s);
    op.print(ss);
    return rust::String::lossy(s);
}

std::unique_ptr<NamedAttribute> NamedAttribute_new(const StringAttr &name, const Attribute &attr)
{
    return std::make_unique<NamedAttribute>(NamedAttribute(name, attr));
}

rust::Str NamedAttribute_getName(const NamedAttribute &attr)
{
    auto ref = attr.getName();
    return rust::Str(ref.data(), ref.size());
}

std::unique_ptr<Attribute> NamedAttribute_getValue(const NamedAttribute &attr)
{
    return std::make_unique<Attribute>(attr.getValue());
}

}
