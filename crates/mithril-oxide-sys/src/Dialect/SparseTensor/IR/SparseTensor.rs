pub use self::ffi::StorageSpecifierType_get;

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/SparseTensor/IR/SparseTensor.hpp");
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        type c_void = crate::c_void;
    }

    #[namespace = "mithril_oxide_sys::sparse_tensor"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/SparseTensor/IR/SparseTensor.hpp");

        #[must_use]
        pub unsafe fn StorageSpecifierType_get(encoding: *const c_void) -> *const c_void;
    }
}
