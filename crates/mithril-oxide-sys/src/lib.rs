#![deny(clippy::pedantic)]
#![deny(warnings)]

use mithril_oxide_sys_proc::codegen;

#[codegen]
mod ffi {
    include!("mlir/InitAllDialects.h");
    include!("mlir/IR/MLIRContext.h");
    include!("mlir/IR/Types.h");
    include!("mlir/IR/Builders.h");
    include!("mlir/IR/Location.h");
    include!("mlir/IR/Block.h");
    include!("mlir/IR/Region.h");
    include!("mlir/IR/Operation.h");
    include!("mlir/Interfaces/DataLayoutInterfaces.h");

    #[codegen(cxx_path = "mlir::MLIRContext::Threading")]
    pub enum Threading {
        DISABLED,
        ENABLED,
    }

    #[codegen(cxx_path = "mlir::MLIRContext", kind = "opaque-sized")]
    pub struct MLIRContext;

    impl MLIRContext {
        #[codegen(cxx_path = "MLIRContext")]
        pub fn new(threading: Threading) -> Self;
        #[codegen(cxx_path = "~MLIRContext")]
        pub fn del(mut self);

        pub fn isMultithreadingEnabled(&mut self) -> bool;

        pub fn loadAllAvailableDialects(&mut self);
    }

    #[codegen(cxx_path = "mlir::registerAllDialects")]
    pub fn registerAllDialects(context: &mut MLIRContext);

    #[codegen(cxx_path = "mlir::Type", kind = "opaque-sized")]
    pub struct Type;

    impl Type {
        pub fn isIndex(&self) -> bool;
        pub fn isFloat8E5M2(&self) -> bool;
        pub fn isFloat8E4M3FN(&self) -> bool;
        pub fn isBF16(&self) -> bool;
        pub fn isF16(&self) -> bool;
        pub fn isF32(&self) -> bool;
        pub fn isF64(&self) -> bool;
        pub fn isF80(&self) -> bool;
        pub fn isF128(&self) -> bool;

        pub fn isInteger(&self, width: u32) -> bool;
        pub fn isSignlessInteger(&self) -> bool;
        #[codegen(cxx_path = "isSignlessInteger")]
        pub fn isSignlessInteger_width(&self, width: u32) -> bool;
        pub fn isUnsignedInteger(&self) -> bool;
        #[codegen(cxx_path = "isUnsignedInteger")]
        pub fn isUnsignedInteger_width(&self, width: u32) -> bool;
        pub fn getIntOrFloatBitWidth(&self) -> u32;

        pub fn isSignlessIntOrIndex(&self) -> bool;
        pub fn isSignlessIntOrIndexOrFloat(&self) -> bool;
        pub fn isSignlessIntOrFloat(&self) -> bool;

        pub fn isIntOrIndex(&self) -> bool;
        pub fn isIntOrFloat(&self) -> bool;
        pub fn isIntOrIndexOrFloat(&self) -> bool;

        pub fn dump(&self);
    }

    #[codegen(cxx_path = "mlir::Location", kind = "opaque-sized")]
    pub struct Location;

    impl Location {
        pub fn dump(&self);
    }

    #[codegen(cxx_path = "mlir::Attribute", kind = "opaque-sized")]
    pub struct Attribute;

    impl Attribute {
        pub fn dump(&self);
    }

    #[codegen(cxx_path = "mlir::Operation", kind = "opaque-sized")]
    pub struct Operation;

    impl Operation {
        //pub fn dump(&self);
    }

    #[codegen(cxx_path = "mlir::ModuleOp", kind = "opaque-sized")]
    pub struct ModuleOp;

    #[codegen(cxx_path = "mlir::Block", kind = "opaque-sized")]
    pub struct Block;

    impl Block {
        //pub fn dump(&self);
    }

    #[codegen(cxx_path = "mlir::Region", kind = "opaque-sized")]
    pub struct Region;

    //impl Region {}

    #[codegen(cxx_path = "mlir::Builder", kind = "opaque-sized")]
    pub struct Builder;

    impl Builder {
        #[codegen(cxx_path = "Builder")]
        pub fn new(context: *mut MLIRContext) -> Self;
    }

    #[codegen(cxx_path = "mlir::DataLayout", kind = "opaque-sized")]
    pub struct DataLayout;

    impl DataLayout {
        #[codegen(cxx_path = "DataLayout")]
        pub fn new(op: ModuleOp) -> Self;

        /// Returns the size of the given type in the current scope.
        pub fn getTypeSize(&self, t: Type) -> u32;

        /// Returns the size in bits of the given type in the current scope.
        pub fn getTypeSizeInBits(&self, t: Type) -> u32;

        /// Returns the required alignment of the given type in the current scope.
        pub fn getTypeABIAlignment(&self, t: Type) -> u32;

        /// Returns the preferred of the given type in the current scope.
        pub fn getTypePreferredAlignment(&self, t: Type) -> u32;
    }
}
