#![deny(clippy::pedantic)]
#![deny(warnings)]

use mithril_oxide_sys_proc::codegen;

// Note: If a method which is known to exist is not being found, ensure that its arguments have the
//   same mutability specifier in both Rust and C++. In other words, arguments which are `*const _`
//   or `&_` must be `const _ *` or `const _ &` respectively; whereas `*mut _` or `&mut _` must be
//   `_ *` or `_ &` respectively.

#[codegen]
pub mod ffi {
    include!("llvm/ADT/StringRef.h");
    include!("mlir/InitAllDialects.h");
    include!("mlir/Interfaces/DataLayoutInterfaces.h");
    include!("mlir/IR/Block.h");
    include!("mlir/IR/Builders.h");
    include!("mlir/IR/BuiltinAttributes.h");
    include!("mlir/IR/Location.h");
    include!("mlir/IR/MLIRContext.h");
    include!("mlir/IR/Operation.h");
    include!("mlir/IR/OperationSupport.h");
    include!("mlir/IR/Region.h");
    include!("mlir/IR/Types.h");
    include!("mlir/IR/Value.h");

    // Manual wrappers (mostly for templated code).
    include!("aux.hpp");

    //
    // LLVM Support Utilities.
    //

    /// A non-owning string reference (aka. Rust's `&str`).
    #[codegen(cxx_path = "llvm::StringRef", kind = "opaque-sized")]
    pub struct StringRef;

    impl StringRef {
        #[codegen(cxx_path = "StringRef")]
        pub fn new(data: *const i8, length: u64) -> Self;
    }

    //
    // MLIR Context.
    //

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
        pub fn allowUnregisteredDialects(&mut self, allow: bool);
        pub fn allowsUnregisteredDialects(&mut self) -> bool;

        pub fn printOpOnDiagnostic(&mut self, enable: bool);
        pub fn shouldPrintStackTraceOnDiagnostic(&mut self) -> bool;
        pub fn printStackTraceOnDiagnostic(&mut self, enable: bool);
    }

    #[codegen(cxx_path = "mlir::registerAllDialects")]
    pub fn registerAllDialects(context: &mut MLIRContext);

    //
    // MLIR Locations.
    //

    /// A generic MLIR location.
    ///
    /// Its concrete counterparts are:
    ///   - CallSiteLoc
    ///   - FileLineColLoc
    ///   - FusedLoc
    ///   - NameLoc
    ///   - OpaqueLoc
    ///   - UnknownLoc
    #[codegen(cxx_path = "mlir::Location", kind = "opaque-sized")]
    pub struct Location;

    impl Location {
        pub fn dump(&self);
    }

    #[codegen(cxx_path = "mlir::UnknownLoc", kind = "opaque-sized")]
    pub struct UnknownLoc;

    impl UnknownLoc {
        pub fn get(context: *mut MLIRContext) -> Self;
    }

    pub fn _UnknownLoc_downgradeTo_Location(value: UnknownLoc) -> Location;

    //
    // MLIR Types.
    //

    /// A generic MLIR type.
    ///
    /// Its concrete counterparts include (but are not limited to):
    ///   - FloatType
    ///   - IndexType
    ///   - IntegerType
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

    //
    // MLIR Attributes.
    //

    /// A generic MLIR attribute.
    #[codegen(cxx_path = "mlir::Attribute", kind = "opaque-sized")]
    pub struct Attribute;

    impl Attribute {
        pub fn dump(&self);
    }

    /// An MLIR string attribute.
    #[codegen(cxx_path = "mlir::StringAttr", kind = "opaque-sized")]
    pub struct StringAttr;

    impl StringAttr {}

    //
    // MLIR Operations.
    //

    /// A generic MLIR operation.
    #[codegen(cxx_path = "mlir::Operation", kind = "opaque-sized")]
    pub struct Operation;

    impl Operation {
        pub fn getBlock(&mut self) -> *mut Block;
        pub fn getContext(&mut self) -> *mut MLIRContext;
        pub fn getParentRegion(&mut self) -> *mut Region;
        pub fn getParentOp(&mut self) -> *mut Operation;
        pub fn getLoc(&mut self) -> Location;
        pub fn setLoc(&mut self, loc: Location);

        pub fn isProperAncestor(&mut self, other: *mut Operation) -> bool;
        pub fn isAncestor(&mut self, other: *mut Operation) -> bool;

        pub fn dump(&mut self);

        pub fn getNumResults(&mut self) -> u32;
        // todo: getresults returns opresult, which inherits value
    }

    /// An operation builder.
    #[codegen(cxx_path = "mlir::OperationState", kind = "opaque-sized")]
    pub struct OperationState;

    impl OperationState {
        #[codegen(cxx_path = "OperationState")]
        pub fn new(location: Location, name: StringRef) -> Self;

        // pub fn addOperands(&mut self, newOperands: ValueRange);
        // pub fn addTypes(&mut self, newTypes: TypeRange);
        pub fn addAttribute(&mut self, name: StringRef, attr: Attribute);
        pub fn addSuccessors(&mut self, successor: *mut Block);
        pub fn addRegion(&mut self) -> *mut Region;
    }

    pub fn _OperationState_addOperands(state: &mut OperationState, data: *mut Value, length: u64);
    pub fn _OperationState_addTypes(state: &mut OperationState, data: *mut Type, length: u64);

    /// The builtin module operation.
    #[codegen(cxx_path = "mlir::ModuleOp", kind = "opaque-sized")]
    pub struct ModuleOp;

    impl ModuleOp {
        pub fn getBodyRegion(&mut self) -> &'static mut Region;

        pub fn setSymNameAttr(&mut self, attr: StringAttr);
        pub fn setSymVisibilityAttr(&mut self, attr: StringAttr);
    }

    pub fn _ModuleOp_create(loc: Location, name: *const StringRef) -> ModuleOp;
    pub fn _ModuleOp_downgradeTo_Operation(value: &mut ModuleOp) -> *mut Operation;

    //
    // Other MLIR stuff.
    //

    /// An MLIR block.
    #[codegen(cxx_path = "mlir::Block", kind = "opaque-sized")]
    pub struct Block;

    impl Block {
        pub fn dump(&mut self);
    }

    /// An MLIR region.
    #[codegen(cxx_path = "mlir::Region", kind = "opaque-sized")]
    pub struct Region;

    impl Region {
        // todo: support default constructors
        // #[codegen(cxx_path = "Region")]
        // pub fn new() -> Self;
        #[codegen(cxx_path = "Region")]
        pub fn new_from_container(container: *mut Operation) -> Self;

        #[codegen(cxx_path = "~Region")]
        pub fn del(mut self);

        pub fn getLoc(&mut self) -> Location;
        pub fn getContext(&mut self) -> *mut MLIRContext;
    }

    /// An MLIR value.
    #[codegen(cxx_path = "mlir::Value", kind = "opaque-sized")]
    pub struct Value;

    impl Value {
        pub fn getParentBlock(&mut self) -> *mut Block;
        pub fn getContext(&self) -> *mut MLIRContext;
        pub fn getParentRegion(&mut self) -> *mut Region;
        pub fn getDefiningOp(&self) -> *mut Operation;
        pub fn getLoc(&self) -> Location;
        pub fn setLoc(&mut self, loc: Location);

        pub fn getType(&self) -> Type;
        pub fn setType(&mut self, ty: Type);

        pub fn dump(&mut self);
    }

    //
    // MLIR Utilities.
    //

    /// An utility to obtain types, attributes...
    #[codegen(cxx_path = "mlir::Builder", kind = "opaque-sized")]
    pub struct Builder;

    impl Builder {
        #[codegen(cxx_path = "Builder")]
        pub fn new(context: *mut MLIRContext) -> Self;
    }

    #[codegen(cxx_path = "mlir::OpBuilder::Listener", kind = "opaque-sized")]
    pub struct OpBuilderListener;

    /// An utility to obtain operations...
    #[codegen(cxx_path = "mlir::OpBuilder", kind = "opaque-sized")]
    pub struct OpBuilder;

    impl OpBuilder {
        pub fn atBlockEnd(block: *mut Block, listener: *mut OpBuilderListener) -> Self;

        pub fn insert(&mut self, op: *mut Operation) -> *mut Operation;
        pub fn create(&mut self, state: &OperationState) -> *mut Operation;
    }

    /// An utility to obtain the size and alignment of MLIR types.
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
