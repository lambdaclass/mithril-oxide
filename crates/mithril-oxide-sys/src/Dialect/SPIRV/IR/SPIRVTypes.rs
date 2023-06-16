#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/SPIRV/IR/SPIRVTypes.hpp");
    }

    #[namespace = "mithril_oxide_sys::spirv"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/SPIRV/IR/SPIRVTypes.hpp");
    }
}
