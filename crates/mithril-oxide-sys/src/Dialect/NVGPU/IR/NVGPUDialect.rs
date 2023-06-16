pub use self::ffi::DeviceAsyncTokenType_get;

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/NVGPU/IR/NVGPUDialect.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        type c_void = crate::c_void;
    }

    #[namespace = "mithril_oxide_sys::nvgpu"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/NVGPU/IR/NVGPUDialect.hpp");

        pub unsafe fn DeviceAsyncTokenType_get(context: *mut MLIRContext) -> *const c_void;
    }
}
