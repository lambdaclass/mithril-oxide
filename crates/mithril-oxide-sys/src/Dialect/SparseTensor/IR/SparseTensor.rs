pub use self::ffi::DeviceAsyncTokenType_get;

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/SparseTensor/IR/SparseTensor.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys::sparse_tensor"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/SparseTensor/IR/SparseTensor.hpp");

        type c_void = crate::IR::Value::ffi::c_void;

        pub unsafe fn StorageSpecifierType_get(encoding: *const c_void) -> *const c_void;
    }
}
