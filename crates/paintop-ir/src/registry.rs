//! The versioned **operation registry** (`IR_SPEC` §6, `plan.md` §6 item 4).
//!
//! The registry is the authoritative, in-memory index from a stable versioned
//! [`OpId`] (`<namespace>.<name>@<major>`) to its [`OperationManifest`]. The
//! compiler consults it to resolve a plan node's `op` field; an agent consults
//! it to discover what operations exist and to look up alternative major
//! versions of a known operation.
//!
//! # What the registry guarantees
//!
//! - **No duplicate registration.** Two manifests with the same [`OpId`] are a
//!   build error, not a silent last-writer-wins: registration rejects the
//!   second with `E_DUPLICATE_OP_ID`.
//! - **Every registered manifest is internally valid.** Registration runs
//!   [`OperationManifest::validate`] so a malformed manifest never enters the
//!   index.
//! - **Distinct lookup failures.** A *missing* operation (`E_OP_NOT_FOUND`) and
//!   a *known operation at an unsupported major* (`E_OP_VERSION_UNSUPPORTED`)
//!   are different, both [`reference`](ErrorClass::Reference) errors, so an
//!   agent can tell "no such op" from "wrong version" and react accordingly.
//!
//! Lookups are by the canonical [`OpId`]. A version mismatch is detected by
//! comparing the requested `(namespace, name)` against the majors actually
//! registered, so the error can report which majors *are* available.
//!
//! ```
//! use paintop_ir::manifest::{
//!     DeterminismTier, OperationManifest, OutputSpec, ResourceKind, RoiCategory, RoiPolicy,
//! };
//! use paintop_ir::registry::OperationRegistry;
//!
//! let manifest = OperationManifest {
//!     id: "filter.invert@1".parse().unwrap(),
//!     impl_version: 1,
//!     summary: String::new(),
//!     determinism: DeterminismTier::Exact,
//!     roi: RoiPolicy { category: RoiCategory::Pointwise, halo_px: None },
//!     inputs: vec![],
//!     outputs: vec![OutputSpec {
//!         name: "image".to_owned(),
//!         kind: ResourceKind::Image,
//!         doc: String::new(),
//!     }],
//!     params: vec![],
//!     implementations: vec!["cpu.reference@1".parse().unwrap()],
//!     test: Default::default(),
//! };
//!
//! let mut registry = OperationRegistry::new();
//! registry.register(manifest).unwrap();
//!
//! let id = "filter.invert@1".parse().unwrap();
//! assert!(registry.get(&id).is_ok());
//!
//! // A different major of a known op is an unsupported-version error, not "not found".
//! let v2 = "filter.invert@2".parse().unwrap();
//! assert_eq!(
//!     registry.get(&v2).unwrap_err().code,
//!     paintop_ir::registry::E_OP_VERSION_UNSUPPORTED,
//! );
//! ```

use std::collections::BTreeMap;

use crate::error::{Error, ErrorClass, Result};
use crate::manifest::{OpId, OperationManifest};

/// A registered operation's canonical id could not be added because an equal id
/// is already present.
pub const E_DUPLICATE_OP_ID: &str = "E_DUPLICATE_OP_ID";

/// No operation with the requested `<namespace>.<name>` is registered at any
/// major version.
pub const E_OP_NOT_FOUND: &str = "E_OP_NOT_FOUND";

/// The requested operation exists at one or more other major versions, but not
/// at the requested major.
pub const E_OP_VERSION_UNSUPPORTED: &str = "E_OP_VERSION_UNSUPPORTED";

/// An in-memory index of operation manifests keyed by their canonical
/// [`OpId`] (`IR_SPEC` §6).
///
/// The index is a [`BTreeMap`] so iteration is deterministic (ordered by
/// namespace, then name, then major), which keeps any derived listing or
/// hash stable across runs — a baseline requirement for the reproducible
/// runtime (`plan.md` §1).
#[derive(Debug, Clone, Default)]
pub struct OperationRegistry {
    by_id: BTreeMap<OpId, OperationManifest>,
}

impl OperationRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a manifest under its canonical [`OpId`].
    ///
    /// The manifest is validated ([`OperationManifest::validate`]) before it is
    /// inserted, so the index only ever holds internally consistent manifests.
    ///
    /// # Errors
    /// - The manifest's own [`schema`](ErrorClass::Schema) error if it fails
    ///   validation.
    /// - A [`reference`](ErrorClass::Reference) error with code
    ///   [`E_DUPLICATE_OP_ID`] if an equal [`OpId`] is already registered.
    pub fn register(&mut self, manifest: OperationManifest) -> Result<()> {
        manifest.validate()?;
        if self.by_id.contains_key(&manifest.id) {
            return Err(Error::new(
                ErrorClass::Reference,
                E_DUPLICATE_OP_ID,
                format!(
                    "operation {} is already registered; ids must be unique within a registry",
                    manifest.id
                ),
            ));
        }
        self.by_id.insert(manifest.id.clone(), manifest);
        Ok(())
    }

    /// Register every manifest from an iterator, stopping at the first failure.
    ///
    /// # Errors
    /// Propagates the first [`register`](Self::register) failure; manifests
    /// registered before the failure remain in the registry.
    pub fn register_all(
        &mut self,
        manifests: impl IntoIterator<Item = OperationManifest>,
    ) -> Result<()> {
        for manifest in manifests {
            self.register(manifest)?;
        }
        Ok(())
    }

    /// Build a registry from an iterator of manifests.
    ///
    /// # Errors
    /// See [`register`](Self::register): the first invalid or duplicate manifest
    /// aborts the load.
    pub fn from_manifests(manifests: impl IntoIterator<Item = OperationManifest>) -> Result<Self> {
        let mut registry = Self::new();
        registry.register_all(manifests)?;
        Ok(registry)
    }

    /// Look up the manifest for a canonical [`OpId`].
    ///
    /// # Errors
    /// - [`E_OP_NOT_FOUND`] ([`reference`](ErrorClass::Reference)) if no
    ///   operation with the requested `<namespace>.<name>` is registered at any
    ///   major version.
    /// - [`E_OP_VERSION_UNSUPPORTED`] ([`reference`](ErrorClass::Reference)) if
    ///   the operation exists at other majors but not the requested one. The
    ///   message reports the available majors.
    pub fn get(&self, id: &OpId) -> Result<&OperationManifest> {
        if let Some(manifest) = self.by_id.get(id) {
            return Ok(manifest);
        }

        let available = self.majors_for(id.namespace(), id.name());
        if available.is_empty() {
            return Err(Error::new(
                ErrorClass::Reference,
                E_OP_NOT_FOUND,
                format!(
                    "no operation {}.{} is registered",
                    id.namespace(),
                    id.name()
                ),
            ));
        }

        let majors = available
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        Err(Error::new(
            ErrorClass::Reference,
            E_OP_VERSION_UNSUPPORTED,
            format!(
                "operation {}.{} is not available at major {}; registered majors: {majors}",
                id.namespace(),
                id.name(),
                id.major(),
            ),
        ))
    }

    /// Whether a manifest with this exact [`OpId`] is registered.
    #[must_use]
    pub fn contains(&self, id: &OpId) -> bool {
        self.by_id.contains_key(id)
    }

    /// The number of registered operation versions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Whether the registry holds no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Iterate over every registered manifest in canonical id order
    /// (namespace, name, major).
    pub fn iter(&self) -> impl Iterator<Item = &OperationManifest> {
        self.by_id.values()
    }

    /// The registered major versions of `<namespace>.<name>`, ascending.
    ///
    /// Empty if the operation is not registered at any major.
    #[must_use]
    pub fn majors_for(&self, namespace: &str, name: &str) -> Vec<u32> {
        self.by_id
            .values()
            .filter(|m| m.id.namespace() == namespace && m.id.name() == name)
            .map(|m| m.id.major())
            .collect()
    }
}

impl<'a> IntoIterator for &'a OperationRegistry {
    type Item = &'a OperationManifest;
    type IntoIter = std::collections::btree_map::Values<'a, OpId, OperationManifest>;

    fn into_iter(self) -> Self::IntoIter {
        self.by_id.values()
    }
}

#[cfg(test)]
mod tests {
    use super::{E_DUPLICATE_OP_ID, E_OP_NOT_FOUND, E_OP_VERSION_UNSUPPORTED, OperationRegistry};
    use crate::error::ErrorClass;
    use crate::manifest::{
        DeterminismTier, OpId, OperationManifest, OutputSpec, ParamSpec, ParamType, ResourceKind,
        RoiCategory, RoiPolicy,
    };

    /// A minimal, valid pointwise manifest for `id`.
    fn manifest(id: &str) -> OperationManifest {
        OperationManifest {
            id: id.parse().unwrap(),
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

    #[test]
    fn register_then_lookup_returns_the_manifest() {
        let mut reg = OperationRegistry::new();
        reg.register(manifest("filter.invert@1")).unwrap();

        let id: OpId = "filter.invert@1".parse().unwrap();
        let found = reg.get(&id).unwrap();
        assert_eq!(found.id, id);
        assert!(reg.contains(&id));
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
    }

    #[test]
    fn duplicate_registration_is_rejected() {
        let mut reg = OperationRegistry::new();
        reg.register(manifest("filter.invert@1")).unwrap();
        let err = reg.register(manifest("filter.invert@1")).unwrap_err();
        assert_eq!(err.class, ErrorClass::Reference);
        assert_eq!(err.code, E_DUPLICATE_OP_ID);
        // The first registration is untouched.
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn duplicate_detection_is_by_full_op_id_not_just_name() {
        // Same namespace/name, different major: both register.
        let reg = OperationRegistry::from_manifests([
            manifest("filter.invert@1"),
            manifest("filter.invert@2"),
        ])
        .unwrap();
        assert_eq!(reg.len(), 2);
        assert!(reg.contains(&"filter.invert@1".parse().unwrap()));
        assert!(reg.contains(&"filter.invert@2".parse().unwrap()));
    }

    #[test]
    fn missing_operation_is_not_found() {
        let reg = OperationRegistry::new();
        let id: OpId = "filter.invert@1".parse().unwrap();
        let err = reg.get(&id).unwrap_err();
        assert_eq!(err.class, ErrorClass::Reference);
        assert_eq!(err.code, E_OP_NOT_FOUND);
    }

    #[test]
    fn known_op_at_wrong_major_is_version_unsupported() {
        let reg = OperationRegistry::from_manifests([
            manifest("filter.invert@1"),
            manifest("filter.invert@3"),
        ])
        .unwrap();

        let id: OpId = "filter.invert@2".parse().unwrap();
        let err = reg.get(&id).unwrap_err();
        assert_eq!(err.class, ErrorClass::Reference);
        assert_eq!(err.code, E_OP_VERSION_UNSUPPORTED);
        // The message reports the actually-registered majors so an agent can
        // pick a supported one.
        assert!(err.message.contains('1'), "{}", err.message);
        assert!(err.message.contains('3'), "{}", err.message);

        // A different name with no registrations is "not found", not "version".
        let other: OpId = "filter.blur@2".parse().unwrap();
        assert_eq!(reg.get(&other).unwrap_err().code, E_OP_NOT_FOUND);
    }

    #[test]
    fn register_validates_manifests_before_inserting() {
        // An invalid manifest (LocalHalo without halo_px) is rejected with the
        // manifest's own schema error and never enters the index.
        let mut bad = manifest("filter.invert@1");
        bad.roi.category = RoiCategory::LocalHalo;
        bad.roi.halo_px = None;

        let mut reg = OperationRegistry::new();
        let err = reg.register(bad).unwrap_err();
        assert_eq!(err.class, ErrorClass::Schema);
        assert_eq!(err.code, "E_ROI_HALO_MISSING");
        assert!(reg.is_empty());
    }

    #[test]
    fn register_all_aborts_on_first_failure_keeping_prior_entries() {
        let mut reg = OperationRegistry::new();
        let err = reg
            .register_all([
                manifest("filter.invert@1"),
                manifest("filter.invert@1"), // duplicate -> abort
                manifest("filter.blur@1"),   // never reached
            ])
            .unwrap_err();
        assert_eq!(err.code, E_DUPLICATE_OP_ID);
        // The first manifest stayed; the post-failure one was not added.
        assert_eq!(reg.len(), 1);
        assert!(reg.contains(&"filter.invert@1".parse().unwrap()));
        assert!(!reg.contains(&"filter.blur@1".parse().unwrap()));
    }

    #[test]
    fn iteration_is_deterministic_by_canonical_id_order() {
        let reg = OperationRegistry::from_manifests([
            manifest("filter.invert@2"),
            manifest("color.convert@1"),
            manifest("filter.invert@1"),
        ])
        .unwrap();
        let ids: Vec<String> = reg.iter().map(|m| m.id.to_string()).collect();
        assert_eq!(
            ids,
            vec![
                "color.convert@1".to_owned(),
                "filter.invert@1".to_owned(),
                "filter.invert@2".to_owned(),
            ]
        );
        // `&registry` IntoIterator agrees.
        let via_ref: Vec<String> = (&reg).into_iter().map(|m| m.id.to_string()).collect();
        assert_eq!(ids, via_ref);
    }

    #[test]
    fn majors_for_reports_registered_versions_ascending() {
        let reg = OperationRegistry::from_manifests([
            manifest("filter.invert@3"),
            manifest("filter.invert@1"),
        ])
        .unwrap();
        assert_eq!(reg.majors_for("filter", "invert"), vec![1, 3]);
        assert!(reg.majors_for("filter", "blur").is_empty());
    }

    #[test]
    fn params_carry_through_registration() {
        // Sanity that a non-trivial manifest survives registration intact.
        let mut m = manifest("filter.invert@1");
        m.params.push(ParamSpec {
            name: "amount".to_owned(),
            ty: ParamType::Float,
            unit: None,
            required: false,
            default: Some(serde_json::json!(1.0)),
            choices: vec![],
            doc: String::new(),
        });
        let reg = OperationRegistry::from_manifests([m.clone()]).unwrap();
        assert_eq!(reg.get(&m.id).unwrap(), &m);
    }
}
