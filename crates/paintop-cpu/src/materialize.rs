//! The `debug.materialize@1` operation (`OP_CATALOG` §1, `plan.md` §18).
//!
//! A resource → the **bit-identical** same resource, plus a materialization
//! barrier the evidence layer can hang an `intermediates/` artifact off.
//!
//! `debug.materialize` is a *semantics-preserving evidence barrier*. It is the
//! one graph node whose entire job is to **not** change anything: it copies its
//! `resource` input to its `resource` output sample-for-sample, byte-for-byte, so
//! that inserting it anywhere in a plan leaves every downstream value — and every
//! downstream content hash — exactly as it was. Its purpose is structural rather
//! than numeric: it forces the runtime to *materialize* the intermediate it sits
//! on (it participates in demand like any other node and produces a concrete
//! output value), which gives the evidence bundle a stable point to write an
//! `intermediates/` artifact for, and which blocks fusion across the barrier so a
//! debugging session can observe the value at that exact point in the graph
//! (`OP_CATALOG` §1, `OP_CATALOG` §18 `graph.barrier`).
//!
//! # Why it must be exact
//!
//! A materialization barrier that perturbed its input — even by a rounding error,
//! a retype, or a normalization — would defeat its own purpose: the downstream
//! hashes would change and the agent could no longer trust that the artifact it
//! captured is the value the rest of the graph actually saw. So
//! `debug.materialize` is [`Exact`](DeterminismTier::Exact) and the *identity*:
//! the output descriptor is the input descriptor unchanged, and the output sample
//! buffer is a verbatim clone of the input sample buffer (`NaN` payloads and all).
//! Its only postcondition is that the round-trip is the identity.
//!
//! # ROI
//!
//! The barrier is [`Pointwise`](RoiCategory::Pointwise): output sample `(x, y)` is
//! input sample `(x, y)`, so a requested output region demands exactly the
//! co-located input region. No halo, no reduction.

use paintop_core::executor::{InputValues, OpImplementation, OutputValues, ResourceValue};
use paintop_ir::{
    AssertionResult, Descriptors, DeterminismTier, Error, ErrorClass, ImplId, InputRegions,
    InputSpec, OpContract, OperationManifest, OutputDescriptors, OutputRegions, OutputSpec,
    ResourceKind, Result, RoiCategory, RoiPolicy, TestMetadata,
};

/// The canonical id of the materialize operation.
pub const MATERIALIZE_OP_ID: &str = "debug.materialize@1";

/// The `resource` input to materialize was absent.
pub const E_MATERIALIZE_INPUT: &str = "E_MATERIALIZE_INPUT";

/// The op produced an output buffer that disagreed with its (identical) input
/// descriptor — an internal invariant violation, since the output is a verbatim
/// clone of the input value.
pub const E_MATERIALIZE_BUFFER: &str = "E_MATERIALIZE_BUFFER";

/// The `debug.materialize@1` operation: a resource → the bit-identical same
/// resource (a semantics-preserving materialization barrier).
#[derive(Debug, Clone, Copy, Default)]
pub struct Materialize;

impl Materialize {
    /// Construct the materialize operation.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// The declared manifest for `debug.materialize@1`.
    ///
    /// # Errors
    /// Propagates the [`schema`](ErrorClass::Schema) error if the hard-coded op /
    /// impl ids are somehow invalid (they are not).
    pub fn manifest() -> Result<OperationManifest> {
        Ok(OperationManifest {
            id: MATERIALIZE_OP_ID.parse()?,
            impl_version: 1,
            summary: "Materialization barrier: pass a resource through bit-identically (the \
                      identity) so the runtime materializes the intermediate and the evidence \
                      bundle can capture it, without changing any downstream value or hash."
                .to_owned(),
            // A verbatim copy: the output is a deterministic function of the input
            // and nothing else.
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                // Output sample (x, y) is input sample (x, y): pure pointwise.
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![InputSpec {
                name: "resource".to_owned(),
                kind: ResourceKind::Image,
                required: true,
                doc: "The resource to materialize. It is passed through unchanged.".to_owned(),
            }],
            outputs: vec![OutputSpec {
                name: "resource".to_owned(),
                kind: ResourceKind::Image,
                doc: "The input resource, bit-identical: same descriptor and same samples."
                    .to_owned(),
            }],
            params: vec![],
            implementations: vec![reference_impl()?],
            test: materialize_test_metadata(),
        })
    }
}

impl OpContract for Materialize {
    fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("resource".to_owned(), ResourceKind::Image)]
    }

    fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
        vec![("resource".to_owned(), ResourceKind::Image)]
    }

    fn infer_outputs(
        &self,
        inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<OutputDescriptors> {
        // The barrier is the identity on types: the output descriptor *is* the
        // input descriptor, unchanged.
        let input = inputs.get("resource").ok_or_else(missing_input)?;
        let mut out = OutputDescriptors::new();
        out.insert("resource".to_owned(), *input);
        Ok(out)
    }

    fn required_inputs(
        &self,
        requested_outputs: &OutputRegions,
        _inputs: &Descriptors,
        _params: &serde_json::Value,
    ) -> Result<InputRegions> {
        // Pointwise: a requested output region needs exactly the co-located input
        // region.
        let mut regions = InputRegions::new();
        if let Some(region) = requested_outputs.get("resource") {
            regions.insert("resource".to_owned(), *region);
        }
        Ok(regions)
    }

    fn validate_postconditions(
        &self,
        outputs: &OutputDescriptors,
        _params: &serde_json::Value,
    ) -> Result<Vec<AssertionResult>> {
        let produced = outputs.contains_key("resource");
        Ok(vec![if produced {
            AssertionResult::pass("produces_resource")
        } else {
            AssertionResult::fail("produces_resource", "no `resource` output produced")
        }])
    }
}

impl OpImplementation for Materialize {
    fn compute(
        &self,
        inputs: &InputValues,
        _params: &serde_json::Value,
    ) -> std::result::Result<OutputValues, Error> {
        let input = inputs.get("resource").ok_or_else(|| {
            Error::new(
                ErrorClass::Reference,
                E_MATERIALIZE_INPUT,
                "debug.materialize requires a `resource` input value".to_owned(),
            )
        })?;

        // The identity: a verbatim clone of the descriptor and every sample. A
        // report resource is carried through by reconstructing the same report
        // value, so a non-raster resource is materialized just as faithfully.
        let output = if let Some(report) = input.as_report() {
            ResourceValue::report(report.clone())
        } else {
            ResourceValue::new(
                *input.descriptor(),
                input.channels(),
                input.samples().to_vec(),
            )
            .map_err(|actual| {
                Error::new(
                    ErrorClass::Execution,
                    E_MATERIALIZE_BUFFER,
                    format!(
                        "debug.materialize produced a buffer of unexpected length {actual}; \
                         the identity copy must match the input descriptor"
                    ),
                )
            })?
        };

        let mut out = OutputValues::new();
        out.insert("resource".to_owned(), output);
        Ok(out)
    }
}

/// The missing-`resource`-input descriptor error.
fn missing_input() -> Error {
    Error::new(
        ErrorClass::Reference,
        E_MATERIALIZE_INPUT,
        "debug.materialize requires a `resource` input".to_owned(),
    )
}

/// The mandatory `cpu.reference@1` oracle implementation id.
fn reference_impl() -> Result<ImplId> {
    ImplId::new("cpu", "reference", 1)
}

/// The verification declarations for `debug.materialize@1`. It is an exact,
/// single-reference identity op, so differential and perceptual do not apply
/// (derived not-applicable); every other category is covered by the
/// analytic-fixture and property tests in this module.
fn materialize_test_metadata() -> TestMetadata {
    use paintop_ir::{CategoryStatus, VerificationCategory, VerificationDeclarations};
    let mut decls = VerificationDeclarations::new();
    for category in [
        VerificationCategory::BuildHygiene,
        VerificationCategory::SchemaContract,
        VerificationCategory::AnalyticFixtures,
        VerificationCategory::PropertyTests,
        VerificationCategory::Metamorphic,
        VerificationCategory::Goldens,
        VerificationCategory::Fuzzing,
        VerificationCategory::Performance,
    ] {
        decls = decls.with(category, CategoryStatus::Covered);
    }
    TestMetadata {
        has_analytic_reference: true,
        has_property_tests: true,
        golden_fixtures: vec![],
        not_applicable_reason: String::new(),
        verification: decls,
    }
}

#[cfg(test)]
mod tests;
