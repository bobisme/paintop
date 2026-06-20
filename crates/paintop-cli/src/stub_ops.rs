//! A small built-in operation registry the CLI ships with.
//!
//! The real MVP operations land in segment 2; until they do, the agent-facing
//! `op list` / `op schema` / `graph` / `validate` surfaces still need a
//! populated [`OperationRegistry`] to read from. This module registers a handful
//! of minimal, internally-valid stub manifests so those commands produce real
//! JSON over real manifests rather than an empty index. Each stub is a genuine
//! [`OperationManifest`] (it passes [`OperationManifest::validate`]); only the
//! executable backend is absent, which is exactly what segment 2 fills in.
//!
//! Reusing the canonical [`OperationManifest`] here — rather than a parallel CLI
//! struct — is the §6.1 rule: the CLI is a thin reader over the IR.

use paintop_ir::{
    DeterminismTier, ImplId, InputSpec, OpId, OperationManifest, OperationRegistry, OutputSpec,
    ParamSpec, ParamType, ParamUnit, ResourceKind, RoiCategory, RoiPolicy, TestMetadata,
};

/// Build the CLI's built-in [`OperationRegistry`] from the stub manifests.
///
/// # Errors
/// Returns the registry's [`schema`](paintop_ir::ErrorClass::Schema) /
/// [`reference`](paintop_ir::ErrorClass::Reference) error if a stub manifest is
/// internally invalid, declares an unparsable id, or two stubs collide on an id.
/// None of these happen for the hand-written stubs here; the fallibility is
/// surfaced rather than unwrapped so the no-unwrap rule holds in library code.
pub fn registry() -> paintop_ir::Result<OperationRegistry> {
    OperationRegistry::from_manifests(manifests()?)
}

/// The stub manifests the CLI registers, in declaration order.
fn manifests() -> paintop_ir::Result<Vec<OperationManifest>> {
    Ok(vec![invert()?, gaussian_blur()?])
}

/// `filter.invert@1`: a pointwise, exact stub with a single image port.
fn invert() -> paintop_ir::Result<OperationManifest> {
    Ok(OperationManifest {
        id: "filter.invert@1".parse()?,
        impl_version: 1,
        summary: "Invert an image's channels (stub manifest; backend lands in segment 2)."
            .to_owned(),
        determinism: DeterminismTier::Exact,
        roi: RoiPolicy {
            category: RoiCategory::Pointwise,
            halo_px: None,
        },
        inputs: vec![image_input("image")],
        outputs: vec![image_output("image")],
        params: vec![],
        implementations: vec![reference_impl()?],
        test: TestMetadata::default(),
    })
}

/// `filter.gaussian_blur@1`: a local-halo, exact stub with a `sigma` parameter.
fn gaussian_blur() -> paintop_ir::Result<OperationManifest> {
    Ok(OperationManifest {
        id: "filter.gaussian_blur@1".parse()?,
        impl_version: 1,
        summary: "Gaussian blur (stub manifest; backend lands in segment 2).".to_owned(),
        determinism: DeterminismTier::Exact,
        roi: RoiPolicy {
            category: RoiCategory::LocalHalo,
            halo_px: Some(3),
        },
        inputs: vec![image_input("image")],
        outputs: vec![image_output("image")],
        params: vec![ParamSpec {
            name: "sigma".to_owned(),
            ty: ParamType::Float,
            unit: Some(ParamUnit::Pixels),
            required: true,
            default: None,
            choices: vec![],
            doc: "Standard deviation of the Gaussian kernel, in pixels.".to_owned(),
        }],
        implementations: vec![reference_impl()?],
        test: TestMetadata::default(),
    })
}

/// A standard single image input port named `name`.
fn image_input(name: &str) -> InputSpec {
    InputSpec {
        name: name.to_owned(),
        kind: ResourceKind::Image,
        required: true,
        doc: String::new(),
    }
}

/// A standard single image output port named `name`.
fn image_output(name: &str) -> OutputSpec {
    OutputSpec {
        name: name.to_owned(),
        kind: ResourceKind::Image,
        doc: String::new(),
    }
}

/// The mandatory `cpu.reference@1` oracle implementation id every op must list.
fn reference_impl() -> paintop_ir::Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// Parse a `<namespace>.<name>@<major>` op id, surfacing the IR parse error.
///
/// # Errors
/// Propagates the [`schema`](paintop_ir::ErrorClass::Schema) `E_INVALID_OP_ID`
/// error if `s` is not a well-formed op id.
pub fn parse_op_id(s: &str) -> paintop_ir::Result<OpId> {
    s.parse()
}

#[cfg(test)]
mod tests {
    use super::registry;

    #[test]
    fn stub_registry_builds_and_is_nonempty() {
        let reg = registry().expect("stub manifests must be internally valid");
        assert!(!reg.is_empty());
        let id = "filter.gaussian_blur@1".parse().expect("valid op id");
        assert!(reg.contains(&id));
    }
}
