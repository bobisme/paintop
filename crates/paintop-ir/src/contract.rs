//! The executable **operation contract** (`IR_SPEC` §18).
//!
//! A manifest ([`OperationManifest`]) only
//! *declares* what an operation is — its ports, params, ROI category, and the
//! implementations it exposes. The runtime behavior of those declarations lives
//! in Rust as an [`OpContract`]: the small, deterministic, *cheap* trio of
//! functions the compiler calls to
//!
//! 1. **infer outputs** — given input descriptors and resolved params, produce
//!    the descriptors of every output port ([`OpContract::infer_outputs`]);
//! 2. **propagate regions backward** — given the output regions a consumer
//!    demands, compute the input regions that must be produced
//!    ([`OpContract::required_inputs`]); this is the executable realization of
//!    the manifest's [`RoiCategory`](crate::manifest::RoiCategory);
//! 3. **validate postconditions** — given the produced outputs, report the
//!    operation's declared invariants as [`AssertionResult`]s
//!    ([`OpContract::validate_postconditions`]).
//!
//! # Manifest ↔ contract consistency
//!
//! The manifest and the contract are two views of the same operation, authored
//! and tested independently, so they can disagree by mistake. The contract
//! re-states its own port names and kinds (via
//! [`OpContract::declared_inputs`]/[`OpContract::declared_outputs`]); a registry
//! that holds both can call [`check_contract_consistency`] to assert the two
//! agree on port names, order, and resource kinds. A disagreement is an
//! [`E_CONTRACT_PORT_MISMATCH`] error — caught at build/test time, never at run
//! time on a live image.
//!
//! These functions must be **deterministic and cheap** (`IR_SPEC` §18): they
//! take descriptors and metadata only, never pixel data, and must not allocate
//! work proportional to image size.

use std::collections::BTreeMap;

use crate::error::{Error, ErrorClass, Result};
use crate::manifest::{OperationManifest, ResourceKind};
use crate::resource::{Rect, ResourceDescriptor};

/// A contract's declared port shape (its name and resource kind) disagreed with
/// the manifest's declaration of the same operation.
pub const E_CONTRACT_PORT_MISMATCH: &str = "E_CONTRACT_PORT_MISMATCH";

/// A name-keyed bag of resource descriptors, used for an operation's *inputs*
/// (`IR_SPEC` §18 `Descriptors`).
///
/// Keyed by port name so the contract can address ports by the names the
/// manifest declares. A [`BTreeMap`] keeps iteration deterministic.
pub type Descriptors = BTreeMap<String, ResourceDescriptor>;

/// A name-keyed bag of resource descriptors, used for an operation's *outputs*
/// (`IR_SPEC` §18 `OutputDescriptors`). Same shape as [`Descriptors`].
pub type OutputDescriptors = BTreeMap<String, ResourceDescriptor>;

/// The output regions a consumer demands, keyed by output port name
/// (`IR_SPEC` §18 `OutputRegions`).
pub type OutputRegions = BTreeMap<String, Rect>;

/// The input regions an operation must consume to satisfy a set of
/// [`OutputRegions`], keyed by input port name (`IR_SPEC` §18 `InputRegions`).
pub type InputRegions = BTreeMap<String, Rect>;

/// The outcome of one declared postcondition / invariant
/// (`IR_SPEC` §18 `AssertionResult`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssertionStatus {
    /// The invariant held.
    Pass,
    /// The invariant was violated.
    Fail,
    /// The invariant does not apply to this operation/configuration and is
    /// recorded as such rather than silently skipped (`AGENT_VERIFICATION` §10:
    /// "not applicable requires a reason").
    NotApplicable,
}

/// The result of evaluating one named postcondition of an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssertionResult {
    /// A stable name identifying the invariant, e.g. `preserves_constant` or
    /// `no_change_outside_mask`.
    pub name: String,
    /// Whether the invariant held, failed, or did not apply.
    pub status: AssertionStatus,
    /// A short human-readable detail, useful when the status is
    /// [`AssertionStatus::Fail`] or [`AssertionStatus::NotApplicable`].
    pub detail: String,
}

impl AssertionResult {
    /// A passing result for the named invariant.
    #[must_use]
    pub fn pass(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: AssertionStatus::Pass,
            detail: String::new(),
        }
    }

    /// A failing result for the named invariant, with a reason.
    #[must_use]
    pub fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: AssertionStatus::Fail,
            detail: detail.into(),
        }
    }

    /// A not-applicable result for the named invariant, with a required reason.
    #[must_use]
    pub fn not_applicable(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: AssertionStatus::NotApplicable,
            detail: reason.into(),
        }
    }
}

/// A typed contract-evaluation error, distinct from the manifest schema errors.
///
/// This is a thin newtype over the central [`Error`] so callers can use `?` in
/// contract code while keeping the agent-facing taxonomy. The contained
/// [`Error`] always carries one of the contract-specific codes (e.g.
/// [`E_CONTRACT_PORT_MISMATCH`]) or a downstream type/semantic code.
pub type ContractError = Error;

/// The executable shape / ROI / postcondition contract of one operation
/// (`IR_SPEC` §18).
///
/// The trait is **object-safe** (it is used behind `dyn OpContract` in the
/// registry) and its methods must be deterministic and cheap. Implementations
/// receive resolved parameters as canonical JSON ([`serde_json::Value`]) so the
/// IR crate does not need to know every operation's concrete param struct.
///
/// `declared_inputs`/`declared_outputs` exist so the contract's port shape can
/// be cross-checked against the manifest by [`check_contract_consistency`]; they
/// must agree with `infer_outputs` (the inferred descriptors' kinds must match
/// the declared output kinds) and with the manifest.
pub trait OpContract {
    /// The contract's declared input ports, as `(name, kind)` in port order.
    ///
    /// Must equal the manifest's `inputs` (name and kind, in order).
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)>;

    /// The contract's declared output ports, as `(name, kind)` in port order.
    ///
    /// Must equal the manifest's `outputs` (name and kind, in order).
    fn declared_outputs(&self) -> Vec<(String, ResourceKind)>;

    /// Infer the descriptor of every output port from the input descriptors and
    /// resolved params (`IR_SPEC` §18 `infer_outputs`).
    ///
    /// Must be deterministic and cheap. The returned map is keyed by output port
    /// name and must cover exactly the declared output ports.
    ///
    /// # Errors
    /// Returns a [`type`](ErrorClass::Type) or [`semantic`](ErrorClass::Semantic)
    /// [`Error`] if the inputs are incompatible with the operation (e.g. a wrong
    /// kind on a port, or an encoding the op forbids).
    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<OutputDescriptors>;

    /// Compute the input regions needed to produce the requested output regions
    /// (`IR_SPEC` §18 `required_inputs`).
    ///
    /// This is the executable realization of the manifest's ROI category: a
    /// pointwise op returns the same region, a blur dilates by its halo, a warp
    /// inverse-transforms the footprint, etc. Must be deterministic and cheap.
    ///
    /// # Errors
    /// Returns a [`type`](ErrorClass::Type) or [`semantic`](ErrorClass::Semantic)
    /// [`Error`] if the requested outputs or inputs are inconsistent with the
    /// operation.
    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        inputs: &Descriptors,
        params: &serde_json::Value,
    ) -> Result<InputRegions>;

    /// Evaluate the operation's declared postconditions over its produced
    /// outputs (`IR_SPEC` §18 `validate_postconditions`).
    ///
    /// Returns one [`AssertionResult`] per declared invariant. Must be
    /// deterministic and cheap; it inspects descriptors and metadata, not bulk
    /// pixel data.
    ///
    /// # Errors
    /// Returns an [`Error`] only if the outputs are structurally malformed (e.g.
    /// a missing declared port), not for an ordinary invariant failure — that is
    /// reported as an [`AssertionStatus::Fail`] result.
    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>>;
}

/// Assert that an [`OpContract`]'s declared ports agree with its manifest.
///
/// Per `IR_SPEC` §18 the manifest declares the ports the executable contract
/// operates over, so the two can be cross-checked.
///
/// The check is exact: input and output port lists must match the manifest on
/// **name, resource kind, and order**. This catches the failure mode where a
/// manifest is edited (a port renamed, a kind changed, a port added/removed)
/// without updating the Rust contract, or vice versa.
///
/// # Errors
/// Returns a [`schema`](ErrorClass::Schema) [`Error`] with code
/// [`E_CONTRACT_PORT_MISMATCH`] on the first disagreement, naming the offending
/// port and which side declared what.
pub fn check_contract_consistency(
    manifest: &OperationManifest,
    contract: &dyn OpContract,
) -> Result<()> {
    let manifest_inputs: Vec<(String, ResourceKind)> = manifest
        .inputs
        .iter()
        .map(|i| (i.name.clone(), i.kind))
        .collect();
    let manifest_outputs: Vec<(String, ResourceKind)> = manifest
        .outputs
        .iter()
        .map(|o| (o.name.clone(), o.kind))
        .collect();

    compare_ports(
        manifest,
        "input",
        &manifest_inputs,
        &contract.declared_inputs(),
    )?;
    compare_ports(
        manifest,
        "output",
        &manifest_outputs,
        &contract.declared_outputs(),
    )?;
    Ok(())
}

/// Compare one side's manifest ports against the contract's declared ports,
/// emitting [`E_CONTRACT_PORT_MISMATCH`] on the first difference.
fn compare_ports(
    manifest: &OperationManifest,
    side: &str,
    declared_by_manifest: &[(String, ResourceKind)],
    declared_by_contract: &[(String, ResourceKind)],
) -> Result<()> {
    let mismatch = |msg: String| {
        Error::new(
            ErrorClass::Schema,
            E_CONTRACT_PORT_MISMATCH,
            format!(
                "operation {}: {side} port contract disagrees: {msg}",
                manifest.id
            ),
        )
    };

    if declared_by_manifest.len() != declared_by_contract.len() {
        return Err(mismatch(format!(
            "manifest declares {} {side} ports but the contract declares {}",
            declared_by_manifest.len(),
            declared_by_contract.len(),
        )));
    }

    for (index, (m, c)) in declared_by_manifest
        .iter()
        .zip(declared_by_contract.iter())
        .enumerate()
    {
        if m.0 != c.0 {
            return Err(mismatch(format!(
                "at {side} port {index} the manifest names {:?} but the contract names {:?}",
                m.0, c.0,
            )));
        }
        if m.1 != c.1 {
            return Err(mismatch(format!(
                "{side} port {:?} is {:?} in the manifest but {:?} in the contract",
                m.0, m.1, c.1,
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        AssertionResult, AssertionStatus, Descriptors, E_CONTRACT_PORT_MISMATCH, InputRegions,
        OpContract, OutputDescriptors, OutputRegions, check_contract_consistency,
    };
    use crate::error::ErrorClass;
    use crate::manifest::{
        DeterminismTier, InputSpec, OperationManifest, OutputSpec, ResourceKind, RoiCategory,
        RoiPolicy, TestMetadata,
    };
    use crate::resource::{
        AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
        Extent, ImageDescriptor, Rect, ResourceDescriptor, ScalarType, SemanticRole,
    };

    /// A minimal pointwise "invert" stub op: one `image` input, one `image`
    /// output, output descriptor equals the input, pointwise ROI.
    struct InvertStub;

    fn image_descriptor() -> ResourceDescriptor {
        ResourceDescriptor::Image(ImageDescriptor {
            extent: Extent::new(64, 48),
            layout: ChannelLayout::Rgba,
            scalar: ScalarType::F32,
            color: ColorEncoding::LinearSrgb,
            range: ColorRange::SceneReferred,
            alpha: AlphaRepresentation::Premultiplied,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        })
    }

    impl OpContract for InvertStub {
        fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }

        fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }

        fn infer_outputs(
            &self,
            inputs: &Descriptors,
            _params: &serde_json::Value,
        ) -> crate::error::Result<OutputDescriptors> {
            let image = inputs.get("image").ok_or_else(|| {
                crate::error::Error::new(
                    ErrorClass::Reference,
                    "E_MISSING_INPUT",
                    "invert stub requires an `image` input",
                )
            })?;
            let mut out = OutputDescriptors::new();
            out.insert("image".to_owned(), *image);
            Ok(out)
        }

        fn required_inputs(
            &self,
            requested_outputs: &OutputRegions,
            _inputs: &Descriptors,
            _params: &serde_json::Value,
        ) -> crate::error::Result<InputRegions> {
            // Pointwise: input region == output region.
            let region = requested_outputs.get("image").copied().ok_or_else(|| {
                crate::error::Error::new(
                    ErrorClass::Reference,
                    "E_MISSING_OUTPUT_REGION",
                    "invert stub requires an `image` output region",
                )
            })?;
            let mut regions = InputRegions::new();
            regions.insert("image".to_owned(), region);
            Ok(regions)
        }

        fn validate_postconditions(
            &self,
            outputs: &OutputDescriptors,
            _params: &serde_json::Value,
        ) -> crate::error::Result<Vec<AssertionResult>> {
            let same_extent = outputs.contains_key("image");
            Ok(vec![if same_extent {
                AssertionResult::pass("output_present")
            } else {
                AssertionResult::fail("output_present", "no `image` output produced")
            }])
        }
    }

    fn invert_manifest() -> OperationManifest {
        OperationManifest {
            id: "filter.invert@1".parse().unwrap(),
            impl_version: 1,
            summary: String::new(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: String::new(),
            }],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: String::new(),
            }],
            params: vec![],
            implementations: vec!["cpu.reference@1".parse().unwrap()],
            test: TestMetadata::default(),
        }
    }

    #[test]
    fn stub_op_infers_outputs_from_inputs() {
        let stub = InvertStub;
        let mut inputs = Descriptors::new();
        inputs.insert("image".to_owned(), image_descriptor());
        let out = stub
            .infer_outputs(&inputs, &serde_json::Value::Null)
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out["image"], image_descriptor());
    }

    #[test]
    fn stub_op_propagates_pointwise_regions() {
        let stub = InvertStub;
        let mut requested = OutputRegions::new();
        requested.insert("image".to_owned(), Rect::new(4, 4, 20, 20));
        let needed = stub
            .required_inputs(&requested, &Descriptors::new(), &serde_json::Value::Null)
            .unwrap();
        // Pointwise: identical region.
        assert_eq!(needed["image"], Rect::new(4, 4, 20, 20));
    }

    #[test]
    fn stub_op_reports_postconditions() {
        let stub = InvertStub;
        let mut outputs = OutputDescriptors::new();
        outputs.insert("image".to_owned(), image_descriptor());
        let results = stub
            .validate_postconditions(&outputs, &serde_json::Value::Null)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].status, AssertionStatus::Pass);

        // With no output, the postcondition fails (it does not error).
        let empty = OutputDescriptors::new();
        let results = stub
            .validate_postconditions(&empty, &serde_json::Value::Null)
            .unwrap();
        assert_eq!(results[0].status, AssertionStatus::Fail);
    }

    #[test]
    fn consistency_passes_when_manifest_and_contract_agree() {
        let manifest = invert_manifest();
        manifest.validate().unwrap();
        check_contract_consistency(&manifest, &InvertStub).unwrap();
    }

    #[test]
    fn consistency_fails_on_renamed_input_port() {
        let mut manifest = invert_manifest();
        manifest.inputs[0].name = "src".to_owned();
        let err = check_contract_consistency(&manifest, &InvertStub).unwrap_err();
        assert_eq!(err.class, ErrorClass::Schema);
        assert_eq!(err.code, E_CONTRACT_PORT_MISMATCH);
        assert!(err.message.contains("input"), "{}", err.message);
    }

    #[test]
    fn consistency_fails_on_changed_output_kind() {
        let mut manifest = invert_manifest();
        manifest.outputs[0].kind = ResourceKind::Mask;
        let err = check_contract_consistency(&manifest, &InvertStub).unwrap_err();
        assert_eq!(err.code, E_CONTRACT_PORT_MISMATCH);
        assert!(err.message.contains("output"), "{}", err.message);
    }

    #[test]
    fn consistency_fails_on_extra_manifest_port() {
        let mut manifest = invert_manifest();
        manifest.inputs.push(InputSpec {
            name: "mask".to_owned(),
            kind: ResourceKind::Mask,
            required: false,
            doc: String::new(),
        });
        let err = check_contract_consistency(&manifest, &InvertStub).unwrap_err();
        assert_eq!(err.code, E_CONTRACT_PORT_MISMATCH);
        assert!(err.message.contains('1') || err.message.contains('2'));
    }

    #[test]
    fn op_contract_is_object_safe() {
        // If this compiles, the trait is object-safe.
        let stub = InvertStub;
        let dynref: &dyn OpContract = &stub;
        assert_eq!(dynref.declared_outputs().len(), 1);
    }
}
