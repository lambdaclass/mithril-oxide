pub use self::ffi::{
    registerAllTranslations, registerFromLLVMIRTranslation, registerFromSPIRVTranslation,
    registerToCppTranslation, registerToLLVMIRTranslation, registerToSPIRVTranslation,
};

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
