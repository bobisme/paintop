//! The **verification category model** and N/A-reason validation
//! (`AGENT_VERIFICATION` §2, §14).
//!
//! `AGENT_VERIFICATION` §2 enumerates the required verification *layers* (build
//! hygiene, schema/contract, analytic fixtures, property tests, metamorphic
//! tests, differential testing, goldens, perceptual evidence, fuzzing, and
//! performance/resource verification). The `verify-op` command (§14, step 10)
//! "fails if any required category is missing", and the manifest "declares which
//! verification categories are applicable" where "not applicable requires a
//! reason" (§14).
//!
//! This module gives that policy a typed home:
//!
//! - [`VerificationCategory`] enumerates the ten layers as a closed set.
//! - [`VerificationCategory::is_applicable`] *derives* applicability from a
//!   manifest (e.g. differential testing only applies once an op declares more
//!   than the lone `cpu.reference` oracle), so applicability is not a free-form
//!   claim an op can over- or under-state.
//! - [`CategoryStatus`] is what a manifest *declares* per category: covered, or
//!   not-applicable-with-a-reason.
//! - [`verify_categories`] is the gate: every *applicable* category must be
//!   declared, an applicable category may only be skipped with a non-empty
//!   reason, and a not-applicable declaration always requires a reason.
//!
//! The model is pure data + a checker; it does not run any test. It is the
//! contract the (later) `verify-op` harness reports against, and the thing the
//! registry can use to reject a manifest that silently skips a layer.
//!
//! ```
//! use paintop_ir::verify::VerificationCategory;
//!
//! // Layers are the closed 0..=9 set in spec order.
//! assert_eq!(VerificationCategory::ALL.len(), 10);
//! assert_eq!(VerificationCategory::Differential.layer(), 5);
//! assert_eq!(VerificationCategory::Differential.as_str(), "differential");
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, ErrorClass, Result};
use crate::manifest::{
    CPU_REFERENCE_BACKEND, CPU_REFERENCE_NAME, DeterminismTier, OperationManifest,
};

/// Raised when an *applicable* verification category is neither covered nor
/// declared not-applicable (`AGENT_VERIFICATION` §14 step 10).
pub const E_VERIFY_CATEGORY_MISSING: &str = "E_VERIFY_CATEGORY_MISSING";
/// Raised when a category is declared **not applicable** without the required
/// reason (`AGENT_VERIFICATION` §14: "not applicable requires a reason").
pub const E_VERIFY_NA_REASON_MISSING: &str = "E_VERIFY_NA_REASON_MISSING";
/// Raised when a manifest declares a status for a category that does **not**
/// apply to it: an unjustified, misleading claim of coverage.
pub const E_VERIFY_CATEGORY_NOT_APPLICABLE: &str = "E_VERIFY_CATEGORY_NOT_APPLICABLE";

/// One required verification layer (`AGENT_VERIFICATION` §2.1–§2.10).
///
/// The set is closed and ordered to mirror the spec's `Layer 0..=9` listing, so
/// exhaustive table tests and deterministic reporting are possible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum VerificationCategory {
    /// Layer 0: build hygiene (`fmt`, `clippy -D warnings`, `test`, `doc`).
    BuildHygiene,
    /// Layer 1: schema and contract tests (manifest/impl agreement, negative
    /// tests from the manifest).
    SchemaContract,
    /// Layer 2: analytic fixtures with a closed-form expected output.
    AnalyticFixtures,
    /// Layer 3: property-based tests (ranges, identities, determinism).
    PropertyTests,
    /// Layer 4: metamorphic tests (relationships under transformed inputs).
    Metamorphic,
    /// Layer 5: differential testing of alternate implementations against the
    /// `cpu.reference` oracle.
    Differential,
    /// Layer 6: golden fixtures.
    Goldens,
    /// Layer 7: perceptual evidence (contact sheets, perceptual metrics).
    Perceptual,
    /// Layer 8: fuzzing and adversarial input.
    Fuzzing,
    /// Layer 9: performance and resource verification.
    Performance,
}

impl VerificationCategory {
    /// Every category, in spec (`Layer 0..=9`) order, for exhaustive iteration
    /// and deterministic reporting.
    pub const ALL: [Self; 10] = [
        Self::BuildHygiene,
        Self::SchemaContract,
        Self::AnalyticFixtures,
        Self::PropertyTests,
        Self::Metamorphic,
        Self::Differential,
        Self::Goldens,
        Self::Perceptual,
        Self::Fuzzing,
        Self::Performance,
    ];

    /// The stable kebab-case wire token (matches the `serde` representation).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BuildHygiene => "build-hygiene",
            Self::SchemaContract => "schema-contract",
            Self::AnalyticFixtures => "analytic-fixtures",
            Self::PropertyTests => "property-tests",
            Self::Metamorphic => "metamorphic",
            Self::Differential => "differential",
            Self::Goldens => "goldens",
            Self::Perceptual => "perceptual",
            Self::Fuzzing => "fuzzing",
            Self::Performance => "performance",
        }
    }

    /// The §2 layer number (0–9) this category corresponds to.
    #[must_use]
    pub const fn layer(self) -> u8 {
        match self {
            Self::BuildHygiene => 0,
            Self::SchemaContract => 1,
            Self::AnalyticFixtures => 2,
            Self::PropertyTests => 3,
            Self::Metamorphic => 4,
            Self::Differential => 5,
            Self::Goldens => 6,
            Self::Perceptual => 7,
            Self::Fuzzing => 8,
            Self::Performance => 9,
        }
    }

    /// Whether this category **applies** to `manifest`, derived from the
    /// manifest's structure rather than self-declared.
    ///
    /// Most layers are universal (every op builds, parses, has properties, can
    /// be fuzzed and benchmarked). Two are structurally conditional:
    ///
    /// - [`Differential`](Self::Differential) applies only when the op declares
    ///   an implementation *other than* the lone `cpu.reference` oracle — there
    ///   is nothing to differentially compare a single reference against.
    /// - [`Perceptual`](Self::Perceptual) applies to ops whose determinism tier
    ///   is not bit-exact ([`Bounded`](DeterminismTier::Bounded) or
    ///   [`Stochastic`](DeterminismTier::Stochastic)): an `exact`/`reproducible`
    ///   op is pinned by numeric goldens, so perceptual evidence is the oracle
    ///   only where exact reproduction is not promised.
    #[must_use]
    pub fn is_applicable(self, manifest: &OperationManifest) -> bool {
        match self {
            Self::Differential => has_non_reference_impl(manifest),
            Self::Perceptual => matches!(
                manifest.determinism,
                DeterminismTier::Bounded | DeterminismTier::Stochastic
            ),
            // The remaining layers apply to every operation.
            Self::BuildHygiene
            | Self::SchemaContract
            | Self::AnalyticFixtures
            | Self::PropertyTests
            | Self::Metamorphic
            | Self::Goldens
            | Self::Fuzzing
            | Self::Performance => true,
        }
    }

    /// The set of categories applicable to `manifest`, in spec order.
    #[must_use]
    pub fn applicable_to(manifest: &OperationManifest) -> Vec<Self> {
        Self::ALL
            .into_iter()
            .filter(|c| c.is_applicable(manifest))
            .collect()
    }
}

/// Whether `manifest` declares any implementation that is not the conventional
/// `cpu.reference` oracle — the condition under which differential testing has
/// something to compare.
fn has_non_reference_impl(manifest: &OperationManifest) -> bool {
    manifest
        .implementations
        .iter()
        .any(|i| !(i.backend() == CPU_REFERENCE_BACKEND && i.name() == CPU_REFERENCE_NAME))
}

/// What a manifest **declares** about one verification category
/// (`AGENT_VERIFICATION` §14).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
#[non_exhaustive]
pub enum CategoryStatus {
    /// The category is covered by tests/evidence.
    Covered,
    /// The category does not apply to this op; the `reason` explains why.
    NotApplicable {
        /// The required justification (`AGENT_VERIFICATION` §14: "not applicable
        /// requires a reason"). Must be non-empty.
        reason: String,
    },
}

impl CategoryStatus {
    /// A `not-applicable` status carrying `reason`.
    #[must_use]
    pub fn not_applicable(reason: impl Into<String>) -> Self {
        Self::NotApplicable {
            reason: reason.into(),
        }
    }

    /// Whether this status is [`Covered`](Self::Covered).
    #[must_use]
    pub const fn is_covered(&self) -> bool {
        matches!(self, Self::Covered)
    }
}

/// A manifest's per-category verification declarations
/// (`AGENT_VERIFICATION` §14).
///
/// A `BTreeMap` keeps the wire form deterministic (categories ordered by their
/// kebab token) so the manifest canonicalizes and hashes stably. An absent entry
/// means *undeclared* — which is a failure for an applicable category and a
/// (harmless) no-op for a non-applicable one.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VerificationDeclarations {
    /// The declared status per category.
    pub by_category: BTreeMap<VerificationCategory, CategoryStatus>,
}

impl VerificationDeclarations {
    /// An empty set of declarations.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare `status` for `category`, replacing any prior declaration.
    #[must_use]
    pub fn with(mut self, category: VerificationCategory, status: CategoryStatus) -> Self {
        self.by_category.insert(category, status);
        self
    }

    /// The declared status for `category`, if any.
    #[must_use]
    pub fn get(&self, category: VerificationCategory) -> Option<&CategoryStatus> {
        self.by_category.get(&category)
    }
}

/// Validate a manifest's verification declarations against its derived
/// applicable categories (`AGENT_VERIFICATION` §14 step 10).
///
/// The rules, evaluated in spec (`Layer 0..=9`) order so the *first* violation
/// is stable:
///
/// 1. Every category that [applies](VerificationCategory::is_applicable) to the
///    manifest must have a declaration. A missing one is
///    [`E_VERIFY_CATEGORY_MISSING`].
/// 2. A category declared [`NotApplicable`](CategoryStatus::NotApplicable) must
///    carry a non-empty `reason` (after trimming whitespace), whether or not the
///    category is derived-applicable. An empty reason is
///    [`E_VERIFY_NA_REASON_MISSING`].
/// 3. A category that does **not** apply may not be declared
///    [`Covered`](CategoryStatus::Covered): claiming coverage of an inapplicable
///    layer is misleading. That is [`E_VERIFY_CATEGORY_NOT_APPLICABLE`]. (An
///    inapplicable category *may* be declared not-applicable, which is
///    redundant but honest, so long as it carries a reason.)
///
/// # Errors
/// Returns a [`semantic`](ErrorClass::Semantic) error with one of the codes
/// above on the first violation.
pub fn verify_categories(
    manifest: &OperationManifest,
    declarations: &VerificationDeclarations,
) -> Result<()> {
    let semantic = |code: &str, msg: String| Error::new(ErrorClass::Semantic, code, msg);

    for category in VerificationCategory::ALL {
        let applicable = category.is_applicable(manifest);
        match declarations.get(category) {
            None => {
                if applicable {
                    return Err(semantic(
                        E_VERIFY_CATEGORY_MISSING,
                        format!(
                            "operation {} does not declare the applicable verification category {:?} \
                             (layer {}); declare it covered or not-applicable with a reason",
                            manifest.id,
                            category.as_str(),
                            category.layer()
                        ),
                    ));
                }
            }
            Some(CategoryStatus::Covered) => {
                if !applicable {
                    return Err(semantic(
                        E_VERIFY_CATEGORY_NOT_APPLICABLE,
                        format!(
                            "operation {} declares verification category {:?} covered, but it does \
                             not apply to this op",
                            manifest.id,
                            category.as_str()
                        ),
                    ));
                }
            }
            Some(CategoryStatus::NotApplicable { reason }) => {
                if reason.trim().is_empty() {
                    return Err(semantic(
                        E_VERIFY_NA_REASON_MISSING,
                        format!(
                            "operation {} declares verification category {:?} not-applicable \
                             without a reason",
                            manifest.id,
                            category.as_str()
                        ),
                    ));
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CategoryStatus, E_VERIFY_CATEGORY_MISSING, E_VERIFY_CATEGORY_NOT_APPLICABLE,
        E_VERIFY_NA_REASON_MISSING, VerificationCategory, VerificationDeclarations,
        verify_categories,
    };
    use crate::error::ErrorClass;
    use crate::manifest::{
        DeterminismTier, ImplId, OperationManifest, OutputSpec, ResourceKind, RoiCategory,
        RoiPolicy,
    };
    use serde_json::json;

    /// A minimal single-reference manifest: differential and perceptual do not
    /// apply (one impl, exact tier).
    fn single_reference_op() -> OperationManifest {
        OperationManifest {
            id: "filter.invert@1".parse().unwrap(),
            impl_version: 1,
            summary: String::new(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: vec![],
            outputs: vec![OutputSpec {
                name: "image".to_owned(),
                kind: ResourceKind::Image,
                doc: String::new(),
            }],
            params: vec![],
            implementations: vec!["cpu.reference@1".parse().unwrap()],
            test: crate::manifest::TestMetadata::default(),
        }
    }

    /// Declarations covering every category that applies to `manifest`.
    fn cover_all_applicable(manifest: &OperationManifest) -> VerificationDeclarations {
        let mut decls = VerificationDeclarations::new();
        for category in VerificationCategory::applicable_to(manifest) {
            decls = decls.with(category, CategoryStatus::Covered);
        }
        decls
    }

    #[test]
    fn category_table_is_exhaustive_and_round_trips() {
        assert_eq!(VerificationCategory::ALL.len(), 10);
        for (i, c) in VerificationCategory::ALL.into_iter().enumerate() {
            assert_eq!(usize::from(c.layer()), i, "layers are 0..=9 in order");
            let token = serde_json::to_value(c).unwrap();
            assert_eq!(token, json!(c.as_str()));
            let back: VerificationCategory = serde_json::from_value(token).unwrap();
            assert_eq!(back, c);
        }
    }

    #[test]
    fn differential_applies_only_with_a_non_reference_impl() {
        let single = single_reference_op();
        assert!(!VerificationCategory::Differential.is_applicable(&single));

        let mut multi = single_reference_op();
        multi
            .implementations
            .push("cpu.simd@1".parse::<ImplId>().unwrap());
        assert!(VerificationCategory::Differential.is_applicable(&multi));
    }

    #[test]
    fn perceptual_applies_only_for_non_exact_tiers() {
        let mut m = single_reference_op();
        for tier in DeterminismTier::ALL {
            m.determinism = tier;
            // a stochastic op needs a seed param to be a valid manifest, but
            // applicability does not depend on that, only on the tier.
            let expect = matches!(tier, DeterminismTier::Bounded | DeterminismTier::Stochastic);
            assert_eq!(
                VerificationCategory::Perceptual.is_applicable(&m),
                expect,
                "{tier:?}"
            );
        }
    }

    #[test]
    fn universal_categories_always_apply() {
        let m = single_reference_op();
        for c in [
            VerificationCategory::BuildHygiene,
            VerificationCategory::SchemaContract,
            VerificationCategory::AnalyticFixtures,
            VerificationCategory::PropertyTests,
            VerificationCategory::Metamorphic,
            VerificationCategory::Goldens,
            VerificationCategory::Fuzzing,
            VerificationCategory::Performance,
        ] {
            assert!(c.is_applicable(&m), "{c:?}");
        }
    }

    #[test]
    fn full_coverage_of_applicable_categories_passes() {
        let m = single_reference_op();
        let decls = cover_all_applicable(&m);
        verify_categories(&m, &decls).unwrap();
    }

    #[test]
    fn skipped_with_reason_passes() {
        // An exact, single-impl op: differential and perceptual do not apply;
        // declare analytic fixtures not-applicable with a reason.
        let m = single_reference_op();
        let mut decls = cover_all_applicable(&m);
        decls = decls.with(
            VerificationCategory::AnalyticFixtures,
            CategoryStatus::not_applicable("output is a pass-through; no analytic form to derive"),
        );
        verify_categories(&m, &decls).unwrap();
    }

    #[test]
    fn missing_applicable_category_fails() {
        let m = single_reference_op();
        let mut decls = cover_all_applicable(&m);
        decls
            .by_category
            .remove(&VerificationCategory::PropertyTests);
        let err = verify_categories(&m, &decls).unwrap_err();
        assert_eq!(err.class, ErrorClass::Semantic);
        assert_eq!(err.code, E_VERIFY_CATEGORY_MISSING);
        assert!(err.message.contains("property-tests"), "{err}");
    }

    #[test]
    fn not_applicable_without_reason_fails() {
        let m = single_reference_op();
        let mut decls = cover_all_applicable(&m);
        decls = decls.with(
            VerificationCategory::Goldens,
            CategoryStatus::not_applicable("   "), // whitespace-only is empty
        );
        let err = verify_categories(&m, &decls).unwrap_err();
        assert_eq!(err.code, E_VERIFY_NA_REASON_MISSING);
        assert!(err.message.contains("goldens"), "{err}");
    }

    #[test]
    fn covered_inapplicable_category_fails() {
        // Differential does not apply to a single-reference op; claiming it
        // covered is misleading and rejected.
        let m = single_reference_op();
        let decls = cover_all_applicable(&m)
            .with(VerificationCategory::Differential, CategoryStatus::Covered);
        let err = verify_categories(&m, &decls).unwrap_err();
        assert_eq!(err.code, E_VERIFY_CATEGORY_NOT_APPLICABLE);
        assert!(err.message.contains("differential"), "{err}");
    }

    #[test]
    fn inapplicable_category_may_be_declared_not_applicable() {
        // Redundant but honest: declaring an inapplicable category n/a with a
        // reason is accepted.
        let m = single_reference_op();
        let decls = cover_all_applicable(&m).with(
            VerificationCategory::Differential,
            CategoryStatus::not_applicable("single reference implementation"),
        );
        verify_categories(&m, &decls).unwrap();
    }

    #[test]
    fn status_serde_round_trips() {
        let covered = CategoryStatus::Covered;
        assert_eq!(
            serde_json::to_value(&covered).unwrap(),
            json!({ "status": "covered" })
        );
        let na = CategoryStatus::not_applicable("because");
        assert_eq!(
            serde_json::to_value(&na).unwrap(),
            json!({ "status": "not-applicable", "reason": "because" })
        );
        let back: CategoryStatus = serde_json::from_value(json!({ "status": "covered" })).unwrap();
        assert_eq!(back, covered);
    }

    #[test]
    fn declarations_serialize_as_a_keyed_map() {
        let decls = VerificationDeclarations::new()
            .with(VerificationCategory::BuildHygiene, CategoryStatus::Covered)
            .with(
                VerificationCategory::Differential,
                CategoryStatus::not_applicable("single impl"),
            );
        let value = serde_json::to_value(&decls).unwrap();
        assert_eq!(value["build-hygiene"], json!({ "status": "covered" }));
        assert_eq!(
            value["differential"],
            json!({ "status": "not-applicable", "reason": "single impl" })
        );
        let back: VerificationDeclarations = serde_json::from_value(value).unwrap();
        assert_eq!(back, decls);
    }
}
