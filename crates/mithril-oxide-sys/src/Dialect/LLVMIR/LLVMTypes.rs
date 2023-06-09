pub use self::ffi::{
    LLVMArrayType_get, LLVMFixedVectorType_get, LLVMFunctionType_get, LLVMMetadataType_get,
    LLVMPPCFP128Type_get, LLVMPointerType_get, LLVMScalableVectorType_get,
    LLVMStructType_getLiteral, LLVMTokenType_get, LLVMVoidType_get, LLVMX86MMXType_get,
};

#[cxx::bridge]
mod ffi {
    #[namespace = "mlir"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/LLVMIR/LLVMTypes.hpp");

        type MLIRContext = crate::IR::MLIRContext::MLIRContext;
    }

    #[namespace = "mithril_oxide_sys"]
    unsafe extern "C++" {
        type c_void = crate::c_void;
    }

    #[namespace = "mithril_oxide_sys::llvm"]
    unsafe extern "C++" {
        include!("mithril-oxide-sys/cpp/Dialect/LLVMIR/LLVMTypes.hpp");

        #[must_use]
        pub unsafe fn LLVMArrayType_get(
            elementType: *const c_void,
            numElements: u32,
        ) -> *const c_void;
        #[must_use]
        pub unsafe fn LLVMFixedVectorType_get(
            elementType: *const c_void,
            numElements: u32,
        ) -> *const c_void;
        #[must_use]
        pub unsafe fn LLVMFunctionType_get(
            result: *const c_void,
            arguments: &[*const c_void],
            isVarArg: bool,
        ) -> *const c_void;
        pub unsafe fn LLVMMetadataType_get(ctx: *mut MLIRContext) -> *const c_void;
        #[must_use]
        pub unsafe fn LLVMPointerType_get(ctx: *const c_void) -> *const c_void;
        pub unsafe fn LLVMPPCFP128Type_get(ctx: *mut MLIRContext) -> *const c_void;
        #[must_use]
        pub unsafe fn LLVMScalableVectorType_get(
            elementType: *const c_void,
            numElements: u32,
        ) -> *const c_void;
        pub unsafe fn LLVMStructType_getLiteral(
            ctx: *mut MLIRContext,
            fields: &[*const c_void],
            isPacked: bool,
        ) -> *const c_void;
        pub unsafe fn LLVMTokenType_get(ctx: *mut MLIRContext) -> *const c_void;
        pub unsafe fn LLVMVoidType_get(ctx: *mut MLIRContext) -> *const c_void;
        pub unsafe fn LLVMX86MMXType_get(ctx: *mut MLIRContext) -> *const c_void;
    }
}
