#include "Attributes.hpp"

#include <memory>
#include <string>

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <rust/cxx.h>


namespace mithril_oxide_sys {

rust::String Attribute_print(const void* attr)
{
    std::string s;
    llvm::raw_string_ostream ss(s);
    Attribute::getFromOpaquePointer(attr).print(ss);
    return rust::String::lossy(s);
}

//std::unique_ptr<NamedAttribute> NamedAttribute_new(const StringAttr &name, const Attribute &attr)
std::unique_ptr<NamedAttribute> NamedAttribute_new(const void* name, const void* attr)
{
    return std::make_unique<NamedAttribute>(
        NamedAttribute(
            StringAttr::getFromOpaquePointer(name),
            Attribute::getFromOpaquePointer(attr)
            ));
}

rust::Str NamedAttribute_getName(const NamedAttribute &attr)
{
    auto ref = attr.getName();
    return rust::Str(ref.data(), ref.size());
}

const void* NamedAttribute_getValue(const NamedAttribute &attr)
{
    return attr.getValue().getAsOpaquePointer();
}

} // namespace mithril_oxide_sys
