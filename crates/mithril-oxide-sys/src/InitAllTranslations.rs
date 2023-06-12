pub use self::ffi::registerAllTranslations;
pub use self::ffi::registerFromLLVMIRTranslation;
pub use self::ffi::registerFromSPIRVTranslation;
pub use self::ffi::registerToCppTranslation;
pub use self::ffi::registerToLLVMIRTranslation;
pub use self::ffi::registerToSPIRVTranslation;

#[cxx::bridge]
pub(crate) mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/InitAllTranslations.hpp");

        pub fn registerFromLLVMIRTranslation();
        pub fn registerFromSPIRVTranslation();
        pub fn registerToCppTranslation();
        pub fn registerToLLVMIRTranslation();
        pub fn registerToSPIRVTranslation();
        pub fn registerAllTranslations();
    }
}
