pub use self::ffi::DeviceAsyncTokenType_get;

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/NVGPU/IR/NVGPUDialect.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys::ml_program"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/NVGPU/IR/NVGPUDialect.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        pub unsafe fn DeviceAsyncTokenType_get(context: *mut MLIRContext) -> *const c_void;
    }
}
