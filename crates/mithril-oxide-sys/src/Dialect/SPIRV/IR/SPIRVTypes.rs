pub use self::ffi::DeviceAsyncTokenType_get;

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/SPIRV/IR/SPIRV.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys::sparse_tensor"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/SPIRV/IR/SPIRV.hpp");

        type c_void = crate::IR::Value::ffi::c_void;
    }
}
