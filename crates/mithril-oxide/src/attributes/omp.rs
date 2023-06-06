use super::Attribute;
use crate::context::Context;
use std::{fmt, marker::PhantomData};

#[derive(Debug)]
pub struct ClauseCancellationConstructTypeAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ClauseCancellationConstructTypeAttr<'c> {}

impl<'c> fmt::Display for ClauseCancellationConstructTypeAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ClauseDependAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ClauseDependAttr<'c> {}

impl<'c> fmt::Display for ClauseDependAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ClauseGrainsizeTypeAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ClauseGrainsizeTypeAttr<'c> {}

impl<'c> fmt::Display for ClauseGrainsizeTypeAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ClauseMemoryOrderKindAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ClauseMemoryOrderKindAttr<'c> {}

impl<'c> fmt::Display for ClauseMemoryOrderKindAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ClauseNumTasksTypeAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ClauseNumTasksTypeAttr<'c> {}

impl<'c> fmt::Display for ClauseNumTasksTypeAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ClauseOrderKindAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ClauseOrderKindAttr<'c> {}

impl<'c> fmt::Display for ClauseOrderKindAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ClauseProcBindKindAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ClauseProcBindKindAttr<'c> {}

impl<'c> fmt::Display for ClauseProcBindKindAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ClauseScheduleKindAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ClauseScheduleKindAttr<'c> {}

impl<'c> fmt::Display for ClauseScheduleKindAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ClauseTaskDependAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ClauseTaskDependAttr<'c> {}

impl<'c> fmt::Display for ClauseTaskDependAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct DeclareTargetAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for DeclareTargetAttr<'c> {}

impl<'c> fmt::Display for DeclareTargetAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct DeclareTargetCaptureClauseAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for DeclareTargetCaptureClauseAttr<'c> {}

impl<'c> fmt::Display for DeclareTargetCaptureClauseAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct DeclareTargetDeviceTypeAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for DeclareTargetDeviceTypeAttr<'c> {}

impl<'c> fmt::Display for DeclareTargetDeviceTypeAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct FlagsAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for FlagsAttr<'c> {}

impl<'c> fmt::Display for FlagsAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct ScheduleModifierAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for ScheduleModifierAttr<'c> {}

impl<'c> fmt::Display for ScheduleModifierAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct TargetAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for TargetAttr<'c> {}

impl<'c> fmt::Display for TargetAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug)]
pub struct VersionAttr<'c> {
    phantom: PhantomData<&'c Context>,
}

impl<'c> Attribute for VersionAttr<'c> {}

impl<'c> fmt::Display for VersionAttr<'c> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}
