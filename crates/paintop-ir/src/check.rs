//! Type, shape, color, and alpha checking over the resolved graph
//! (`plan.md` §10.1 phase 4, `IR_SPEC` §7, §18).
//!
//! Phase 3 ([`resolve_plan`](crate::resolve::resolve_plan)) produces a
//! [`ResolvedGraph`]: every reference resolves, every required port is wired,
//! and the graph is an acyclic, topologically ordered DAG. This module is
//! phase 4. It walks the resolved graph in topological order and, for each node,
//!
//! 1. assembles the concrete [`ResourceDescriptor`] flowing into every wired
//!    input port — from the supplied external input descriptors for `input:`
//!    edges, and from already-inferred upstream outputs for `node:` edges;
//! 2. checks each input descriptor's *kind* against the port's declared
//!    [`ResourceKind`] (the type/shape gate), since a node may not feed a mask
//!    into an `Image` port;
//! 3. invokes the operation's executable [`OpContract::infer_outputs`], which is
//!    where the color-encoding, alpha-representation, semantic-role, and extent
//!    rules live — an op rejects an `srgb` image where it needs linear light, a
//!    `straight` alpha where it needs premultiplied, a `depth` field where it
//!    needs color, with a [`type`](ErrorClass::Type) or
//!    [`semantic`](ErrorClass::Semantic) error;
//! 4. records every inferred output descriptor so downstream nodes and the
//!    exports can be checked against it.
//!
//! The result is a [`CheckedGraph`]: the resolved graph annotated with a
//! concrete inferred descriptor for every node output and every export, on which
//! later phases (policy, ROI analysis) build.
//!
//! # Where input descriptors come from
//!
//! The checker is pure over descriptors: it does not decode files. The concrete
//! descriptor of each `input:` resource is supplied by the caller (the input
//! loader resolves an [`InputDecl`](crate::plan::InputDecl) to a descriptor in a
//! later bone). This keeps the checker deterministic, cheap, and testable
//! against synthetic descriptors.
//!
//! # The contract registry
//!
//! [`OperationRegistry`] holds the manifests
//! (data); the executable [`OpContract`]s (code) live in a separate
//! [`ContractRegistry`] so the IR crate need not link every op implementation.
//! [`ContractRegistry::check_consistency`] cross-checks the two so a manifest
//! and its contract can never silently disagree on ports.

use std::collections::BTreeMap;

use crate::contract::{Descriptors, OpContract, check_contract_consistency};
use crate::error::{Error, ErrorClass, ErrorContext, Result};
use crate::manifest::{OpId, ResourceKind};
use crate::plan::Plan;
use crate::registry::OperationRegistry;
use crate::resolve::{Reference, ResolvedGraph, ResolvedNode};
use crate::resource::ResourceDescriptor;

/// A node was fed the wrong resource kind.
///
/// The descriptor flowing into an input port does not match the port's declared
/// [`ResourceKind`] (e.g. a `Mask` flowing into an `Image` port). This is the
/// type/shape gate, run before an operation's own color/alpha checks.
pub const E_PORT_KIND_MISMATCH: &str = "E_PORT_KIND_MISMATCH";

/// An `input:` reference named a resource whose concrete descriptor the caller
/// did not supply, so the checker cannot type the edge.
pub const E_MISSING_INPUT_DESCRIPTOR: &str = "E_MISSING_INPUT_DESCRIPTOR";

/// A contract omitted a declared output descriptor.
///
/// An operation's [`infer_outputs`](OpContract::infer_outputs) did not produce a
/// descriptor for a port it (and its manifest) declares as an output — a
/// contract bug surfaced as a type error rather than a later panic.
pub const E_OUTPUT_PORT_NOT_INFERRED: &str = "E_OUTPUT_PORT_NOT_INFERRED";

/// No executable [`OpContract`] is registered for an operation that the resolved
/// graph uses. The manifest exists (resolution passed) but its contract is
/// missing, so its outputs cannot be inferred.
pub const E_CONTRACT_NOT_FOUND: &str = "E_CONTRACT_NOT_FOUND";

/// An in-memory index from an [`OpId`] to its executable [`OpContract`]
/// (`IR_SPEC` §18).
///
/// Distinct from the manifest [`OperationRegistry`]: manifests are data the
/// agent reads; contracts are the deterministic Rust functions the checker
/// calls. Keeping them separate lets the IR crate hold manifests without linking
/// every operation implementation, while [`check_consistency`](Self::check_consistency)
/// guarantees the two never disagree on ports.
#[derive(Default)]
pub struct ContractRegistry {
    by_id: BTreeMap<OpId, Box<dyn OpContract>>,
}

impl ContractRegistry {
    /// Create an empty contract registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the executable contract for `id`.
    ///
    /// # Errors
    /// Returns a [`schema`](ErrorClass::Schema) [`Error`] with code
    /// [`E_DUPLICATE_CONTRACT`] if a contract is already registered for `id`.
    pub fn register(&mut self, id: OpId, contract: Box<dyn OpContract>) -> Result<()> {
        if self.by_id.contains_key(&id) {
            return Err(Error::new(
                ErrorClass::Schema,
                E_DUPLICATE_CONTRACT,
                format!("a contract is already registered for operation {id}"),
            ));
        }
        self.by_id.insert(id, contract);
        Ok(())
    }

    /// Look up the contract for `id`.
    #[must_use]
    pub fn get(&self, id: &OpId) -> Option<&dyn OpContract> {
        self.by_id.get(id).map(Box::as_ref)
    }

    /// Whether a contract is registered for `id`.
    #[must_use]
    pub fn contains(&self, id: &OpId) -> bool {
        self.by_id.contains_key(id)
    }

    /// The number of registered contracts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Whether the registry holds no contracts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Cross-check every registered contract against its manifest in `manifests`,
    /// asserting they agree on input/output port names, kinds, and order
    /// (`IR_SPEC` §18).
    ///
    /// # Errors
    /// - [`E_CONTRACT_NOT_FOUND`] ([`reference`](ErrorClass::Reference)) if a
    ///   contract is registered for an `id` that `manifests` does not know.
    /// - [`E_CONTRACT_PORT_MISMATCH`](crate::contract::E_CONTRACT_PORT_MISMATCH)
    ///   ([`schema`](ErrorClass::Schema)) on the first manifest/contract port
    ///   disagreement.
    pub fn check_consistency(&self, manifests: &OperationRegistry) -> Result<()> {
        for (id, contract) in &self.by_id {
            let manifest = manifests.get(id).map_err(|_| {
                Error::new(
                    ErrorClass::Reference,
                    E_CONTRACT_NOT_FOUND,
                    format!("contract registered for {id} but no manifest is registered for it"),
                )
            })?;
            check_contract_consistency(manifest, contract.as_ref())?;
        }
        Ok(())
    }
}

/// A contract was registered twice for the same [`OpId`].
pub const E_DUPLICATE_CONTRACT: &str = "E_DUPLICATE_CONTRACT";

/// A type/shape/color/alpha-checked view over a [`ResolvedGraph`]
/// (`plan.md` §10.1 phase 4).
///
/// Every node output and every export has a concrete inferred
/// [`ResourceDescriptor`] whose kind matches its declaration and whose
/// color/alpha/semantic descriptor was accepted by the producing operation's
/// contract.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckedGraph {
    /// Per-node inferred output descriptors, keyed by node id then output port.
    outputs: BTreeMap<String, BTreeMap<String, ResourceDescriptor>>,
    /// The concrete descriptor of each external `input:` resource, keyed by input
    /// name — the same map [`check_graph`] type-checked the graph against. Retained
    /// so a downstream pass (notably the backward ROI analysis) can drive an
    /// operation's [`required_inputs`](crate::contract::OpContract::required_inputs)
    /// for a node reading an external input, which needs that input's descriptor.
    inputs: BTreeMap<String, ResourceDescriptor>,
    /// Inferred descriptor for each export, in the resolved graph's export order.
    exports: Vec<(String, ResourceDescriptor)>,
}

impl CheckedGraph {
    /// The inferred descriptor of node `node`'s output port `port`, if any.
    #[must_use]
    pub fn output(&self, node: &str, port: &str) -> Option<&ResourceDescriptor> {
        self.outputs.get(node).and_then(|ports| ports.get(port))
    }

    /// The concrete descriptor of the external `input:<input>` resource, if the
    /// graph declared one. The same map [`check_graph`] received and validated.
    #[must_use]
    pub fn input(&self, input: &str) -> Option<&ResourceDescriptor> {
        self.inputs.get(input)
    }

    /// Every inferred output descriptor of `node`, keyed by output port name.
    #[must_use]
    pub fn node_outputs(&self, node: &str) -> Option<&BTreeMap<String, ResourceDescriptor>> {
        self.outputs.get(node)
    }

    /// The inferred descriptors of every export, in export-id order.
    #[must_use]
    pub fn exports(&self) -> &[(String, ResourceDescriptor)] {
        &self.exports
    }
}

/// Type-, shape-, color-, and alpha-check a [`ResolvedGraph`], inferring a
/// concrete descriptor for every node output and export (`plan.md` §10.1
/// phase 4).
///
/// `inputs` supplies the concrete descriptor of each `input:` resource (resolved
/// from the plan's [`InputDecl`](crate::plan::InputDecl)s by the input loader).
/// `contracts` supplies the executable [`OpContract`] for every operation the
/// graph uses; the checker calls [`infer_outputs`](OpContract::infer_outputs) in
/// topological order so every upstream output is available when a node is
/// checked.
///
/// The checker performs the kind gate (an `Image` port may not be fed a `Mask`)
/// and then delegates the color-encoding, alpha-representation, semantic-role,
/// and extent rules to the operation's contract, which surfaces them as
/// [`type`](ErrorClass::Type) / [`semantic`](ErrorClass::Semantic) errors.
///
/// # Errors
/// - [`E_MISSING_INPUT_DESCRIPTOR`] ([`type`](ErrorClass::Type)) if an `input:`
///   edge names a resource not present in `inputs`.
/// - [`E_CONTRACT_NOT_FOUND`] ([`reference`](ErrorClass::Reference)) if a node's
///   operation has no registered contract.
/// - [`E_PORT_KIND_MISMATCH`] ([`type`](ErrorClass::Type)) if a descriptor's
///   kind does not match the input port's declared [`ResourceKind`].
/// - any [`type`](ErrorClass::Type) / [`semantic`](ErrorClass::Semantic) error
///   the operation's contract raises (color/alpha/semantic/extent mismatches).
/// - [`E_OUTPUT_PORT_NOT_INFERRED`] ([`type`](ErrorClass::Type)) if a contract
///   omits a declared output descriptor.
pub fn check_graph(
    plan: &Plan,
    graph: &ResolvedGraph,
    manifests: &OperationRegistry,
    contracts: &ContractRegistry,
    inputs: &BTreeMap<String, ResourceDescriptor>,
) -> Result<CheckedGraph> {
    // The resolved graph does not retain node params (they are not part of the
    // wiring); index them back from the plan so param-dependent inference (a
    // resize's target extent, a fill's encoding) sees them.
    let params_by_node: BTreeMap<&str, &serde_json::Map<String, serde_json::Value>> = plan
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), &n.params))
        .collect();

    let mut node_outputs: BTreeMap<String, BTreeMap<String, ResourceDescriptor>> = BTreeMap::new();

    // Topological order guarantees every upstream node was already inferred.
    for node_id in graph.topological_order() {
        let Some(node) = graph.node(node_id) else {
            continue;
        };
        let outputs = check_node(
            node,
            manifests,
            contracts,
            inputs,
            &node_outputs,
            &params_by_node,
        )?;
        node_outputs.insert(node_id.clone(), outputs);
    }

    // Resolve each export's descriptor from the inferred outputs / inputs.
    let mut exports = Vec::with_capacity(graph.exports().len());
    for export in graph.exports() {
        let descriptor =
            resolve_reference(&export.resource, inputs, &node_outputs).map_err(|err| {
                err.with_context(
                    ErrorContext::default().with_path(format!("/exports/{}/resource", export.id)),
                )
            })?;
        exports.push((export.id.clone(), descriptor));
    }

    Ok(CheckedGraph {
        outputs: node_outputs,
        inputs: inputs.clone(),
        exports,
    })
}

/// Check one resolved node: type-gate its inputs, infer its outputs, and verify
/// the contract produced every declared output port.
fn check_node(
    node: &ResolvedNode,
    manifests: &OperationRegistry,
    contracts: &ContractRegistry,
    inputs: &BTreeMap<String, ResourceDescriptor>,
    node_outputs: &BTreeMap<String, BTreeMap<String, ResourceDescriptor>>,
    params_by_node: &BTreeMap<&str, &serde_json::Map<String, serde_json::Value>>,
) -> Result<BTreeMap<String, ResourceDescriptor>> {
    let manifest = manifests.get(&node.op)?;
    let contract = contracts.get(&node.op).ok_or_else(|| {
        Error::new(
            ErrorClass::Reference,
            E_CONTRACT_NOT_FOUND,
            format!(
                "node {:?} uses operation {} but no executable contract is registered for it",
                node.id, node.op
            ),
        )
        .with_context(ErrorContext::default().with_node(node.id.clone()))
    })?;

    // Assemble and type-gate the input descriptors flowing into each wired port.
    let mut input_descriptors: Descriptors = Descriptors::new();
    for (port, reference) in &node.inputs {
        let descriptor = resolve_reference(reference, inputs, node_outputs).map_err(|err| {
            err.with_context(
                ErrorContext::default()
                    .with_node(node.id.clone())
                    .with_path(format!("/nodes/{}/in/{port}", node.id)),
            )
        })?;

        // Kind gate: the descriptor's kind must match the declared port kind.
        let expected = declared_input_kind(manifest, port);
        if let Some(expected) = expected {
            let actual = descriptor.kind();
            if actual != expected {
                return Err(Error::new(
                    ErrorClass::Type,
                    E_PORT_KIND_MISMATCH,
                    format!(
                        "node {:?} input port {port:?} of {} expects a {expected:?} but is fed a \
                         {actual:?}",
                        node.id, node.op
                    ),
                )
                .with_context(
                    ErrorContext::default()
                        .with_node(node.id.clone())
                        .with_path(format!("/nodes/{}/in/{port}", node.id))
                        .with_actual(format!("{actual:?}"))
                        .with_expected(format!("{expected:?}")),
                ));
            }
        }

        input_descriptors.insert(port.clone(), descriptor);
    }

    // Delegate the color/alpha/semantic/extent rules to the operation contract,
    // passing the node's canonical params for param-dependent inference.
    let params = params_by_node
        .get(node.id.as_str())
        .map_or_else(serde_json::Value::default, |params| {
            serde_json::Value::Object((*params).clone())
        });
    let inferred = contract
        .infer_outputs(&input_descriptors, &params)
        .map_err(|err| attribute_to_node(err, &node.id))?;

    // The contract must produce a descriptor for every manifest-declared output.
    let mut outputs = BTreeMap::new();
    for spec in &manifest.outputs {
        let descriptor = inferred.get(&spec.name).ok_or_else(|| {
            Error::new(
                ErrorClass::Type,
                E_OUTPUT_PORT_NOT_INFERRED,
                format!(
                    "operation {} did not infer a descriptor for its declared output port {:?}",
                    node.op, spec.name
                ),
            )
            .with_context(ErrorContext::default().with_node(node.id.clone()))
        })?;
        outputs.insert(spec.name.clone(), *descriptor);
    }

    Ok(outputs)
}

/// Resolve a [`Reference`] to the concrete descriptor it carries: an external
/// input from `inputs`, or an upstream node output from `node_outputs`.
fn resolve_reference(
    reference: &Reference,
    inputs: &BTreeMap<String, ResourceDescriptor>,
    node_outputs: &BTreeMap<String, BTreeMap<String, ResourceDescriptor>>,
) -> Result<ResourceDescriptor> {
    match reference {
        Reference::Input { input } => inputs.get(input).copied().ok_or_else(|| {
            Error::new(
                ErrorClass::Type,
                E_MISSING_INPUT_DESCRIPTOR,
                format!(
                    "input {input:?} has no resolved descriptor; supply it before type checking"
                ),
            )
            .with_context(ErrorContext::default().with_actual(format!("input:{input}")))
        }),
        Reference::Node { node, port } => node_outputs
            .get(node)
            .and_then(|ports| ports.get(port))
            .copied()
            .ok_or_else(|| {
                // Resolution already proved the node/port exist, so a miss here
                // means topological inference did not reach the producer.
                Error::new(
                    ErrorClass::Type,
                    E_OUTPUT_PORT_NOT_INFERRED,
                    format!("output {node:?}/{port:?} was referenced before it was inferred"),
                )
            }),
    }
}

/// The declared [`ResourceKind`] of input port `port` on `manifest`, if it
/// declares one.
fn declared_input_kind(
    manifest: &crate::manifest::OperationManifest,
    port: &str,
) -> Option<ResourceKind> {
    manifest
        .inputs
        .iter()
        .find(|i| i.name == port)
        .map(|i| i.kind)
}

/// Attach a node id to an error if it does not already carry one.
fn attribute_to_node(mut err: Error, node_id: &str) -> Error {
    if err.context.node.is_none() {
        let context = (*err.context).clone().with_node(node_id.to_owned());
        err = err.with_context(context);
    }
    err
}

#[cfg(test)]
mod tests {
    use super::{
        CheckedGraph, ContractRegistry, E_CONTRACT_NOT_FOUND, E_DUPLICATE_CONTRACT,
        E_MISSING_INPUT_DESCRIPTOR, E_PORT_KIND_MISMATCH, check_graph,
    };
    use crate::contract::{
        AssertionResult, Descriptors, InputRegions, OpContract, OutputDescriptors, OutputRegions,
    };
    use crate::error::{Error, ErrorClass, Result};
    use crate::manifest::{
        DeterminismTier, InputSpec, OperationManifest, OutputSpec, ResourceKind, RoiCategory,
        RoiPolicy, TestMetadata,
    };
    use crate::plan::{Plan, parse_plan};
    use crate::registry::OperationRegistry;
    use crate::resolve::resolve_plan;
    use crate::resource::{
        AlphaRepresentation, ChannelLayout, ColorEncoding, ColorRange, CoordinateConvention,
        Extent, ImageDescriptor, MaskDescriptor, MaskMeaning, ResourceDescriptor, ScalarType,
        SemanticRole, ValidRange,
    };
    use std::collections::BTreeMap;

    // ---- Synthetic descriptors --------------------------------------------

    fn image(color: ColorEncoding, alpha: AlphaRepresentation) -> ResourceDescriptor {
        ResourceDescriptor::Image(ImageDescriptor {
            extent: Extent::new(64, 48),
            layout: ChannelLayout::Rgba,
            scalar: ScalarType::F32,
            color,
            range: ColorRange::SceneReferred,
            alpha,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
            semantic: SemanticRole::Color,
        })
    }

    fn linear_premul() -> ResourceDescriptor {
        image(
            ColorEncoding::LinearSrgb,
            AlphaRepresentation::Premultiplied,
        )
    }

    fn mask() -> ResourceDescriptor {
        ResourceDescriptor::Mask(MaskDescriptor {
            extent: Extent::new(64, 48),
            scalar: ScalarType::F32,
            range: ValidRange::Bounded { min: 0.0, max: 1.0 },
            meaning: MaskMeaning::Coverage,
            coordinates: CoordinateConvention::PixelCenterUpperLeft,
        })
    }

    // ---- Stub manifests / contracts ---------------------------------------

    fn op(
        id: &str,
        inputs: &[(&str, ResourceKind)],
        outputs: &[(&str, ResourceKind)],
    ) -> OperationManifest {
        OperationManifest {
            id: id.parse().unwrap(),
            impl_version: 1,
            summary: String::new(),
            determinism: DeterminismTier::Exact,
            roi: RoiPolicy {
                category: RoiCategory::Pointwise,
                halo_px: None,
            },
            inputs: inputs
                .iter()
                .map(|(name, kind)| InputSpec {
                    name: (*name).to_owned(),
                    kind: *kind,
                    required: true,
                    doc: String::new(),
                })
                .collect(),
            outputs: outputs
                .iter()
                .map(|(name, kind)| OutputSpec {
                    name: (*name).to_owned(),
                    kind: *kind,
                    doc: String::new(),
                })
                .collect(),
            params: vec![],
            implementations: vec!["cpu.reference@1".parse().unwrap()],
            test: TestMetadata::default(),
        }
    }

    /// A source op: no inputs, one `image` output fixed to a supplied descriptor.
    struct SourceStub(ResourceDescriptor);
    impl OpContract for SourceStub {
        fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
            vec![]
        }
        fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn infer_outputs(
            &self,
            _inputs: &Descriptors,
            _params: &serde_json::Value,
        ) -> Result<OutputDescriptors> {
            let mut out = OutputDescriptors::new();
            out.insert("image".to_owned(), self.0);
            Ok(out)
        }
        fn required_inputs(
            &self,
            _o: &OutputRegions,
            _i: &Descriptors,
            _p: &serde_json::Value,
        ) -> Result<InputRegions> {
            Ok(InputRegions::new())
        }
        fn validate_postconditions(
            &self,
            _o: &OutputDescriptors,
            _p: &serde_json::Value,
        ) -> Result<Vec<AssertionResult>> {
            Ok(vec![])
        }
    }

    /// A blur stub that *requires* a linear-light, premultiplied `image` input
    /// and passes its descriptor through. The color/alpha policy lives here, in
    /// the contract, exactly as the spec intends.
    struct LinearBlurStub;
    impl OpContract for LinearBlurStub {
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
        ) -> Result<OutputDescriptors> {
            let image = inputs.get("image").ok_or_else(|| {
                Error::new(ErrorClass::Type, "E_MISSING_INPUT", "blur needs an `image`")
            })?;
            let ResourceDescriptor::Image(desc) = image else {
                return Err(Error::new(
                    ErrorClass::Type,
                    "E_WRONG_KIND",
                    "blur needs an Image",
                ));
            };
            if !desc.color.is_linear_light() {
                return Err(Error::new(
                    ErrorClass::Semantic,
                    "E_COLOR_ENCODING_MISMATCH",
                    "filter.blur requires a linear image",
                )
                .with_context(
                    crate::error::ErrorContext::default()
                        .with_actual(format!("{:?}", desc.color))
                        .with_expected("linear-*"),
                ));
            }
            if desc.alpha != AlphaRepresentation::Premultiplied {
                return Err(Error::new(
                    ErrorClass::Semantic,
                    "E_ALPHA_REPRESENTATION_MISMATCH",
                    "filter.blur requires premultiplied alpha",
                ));
            }
            let mut out = OutputDescriptors::new();
            out.insert("image".to_owned(), *image);
            Ok(out)
        }
        fn required_inputs(
            &self,
            o: &OutputRegions,
            _i: &Descriptors,
            _p: &serde_json::Value,
        ) -> Result<InputRegions> {
            let mut regions = InputRegions::new();
            if let Some(region) = o.get("image") {
                regions.insert("image".to_owned(), *region);
            }
            Ok(regions)
        }
        fn validate_postconditions(
            &self,
            _o: &OutputDescriptors,
            _p: &serde_json::Value,
        ) -> Result<Vec<AssertionResult>> {
            Ok(vec![])
        }
    }

    /// A masked-replace stub: `image` + `mask` -> `image`, both required, output
    /// equals the `image` input.
    struct MaskedReplaceStub;
    impl OpContract for MaskedReplaceStub {
        fn declared_inputs(&self) -> Vec<(String, ResourceKind)> {
            vec![
                ("image".to_owned(), ResourceKind::Image),
                ("mask".to_owned(), ResourceKind::Mask),
            ]
        }
        fn declared_outputs(&self) -> Vec<(String, ResourceKind)> {
            vec![("image".to_owned(), ResourceKind::Image)]
        }
        fn infer_outputs(
            &self,
            inputs: &Descriptors,
            _params: &serde_json::Value,
        ) -> Result<OutputDescriptors> {
            let image = inputs
                .get("image")
                .copied()
                .ok_or_else(|| Error::new(ErrorClass::Type, "E_MISSING_INPUT", "needs `image`"))?;
            let mut out = OutputDescriptors::new();
            out.insert("image".to_owned(), image);
            Ok(out)
        }
        fn required_inputs(
            &self,
            _o: &OutputRegions,
            _i: &Descriptors,
            _p: &serde_json::Value,
        ) -> Result<InputRegions> {
            Ok(InputRegions::new())
        }
        fn validate_postconditions(
            &self,
            _o: &OutputDescriptors,
            _p: &serde_json::Value,
        ) -> Result<Vec<AssertionResult>> {
            Ok(vec![])
        }
    }

    fn registry() -> OperationRegistry {
        OperationRegistry::from_manifests([
            op("source.create@1", &[], &[("image", ResourceKind::Image)]),
            op(
                "filter.blur@1",
                &[("image", ResourceKind::Image)],
                &[("image", ResourceKind::Image)],
            ),
            op(
                "composite.masked_replace@1",
                &[("image", ResourceKind::Image), ("mask", ResourceKind::Mask)],
                &[("image", ResourceKind::Image)],
            ),
        ])
        .unwrap()
    }

    fn contracts(source: ResourceDescriptor) -> ContractRegistry {
        let mut c = ContractRegistry::new();
        c.register(
            "source.create@1".parse().unwrap(),
            Box::new(SourceStub(source)),
        )
        .unwrap();
        c.register("filter.blur@1".parse().unwrap(), Box::new(LinearBlurStub))
            .unwrap();
        c.register(
            "composite.masked_replace@1".parse().unwrap(),
            Box::new(MaskedReplaceStub),
        )
        .unwrap();
        c
    }

    fn no_inputs() -> BTreeMap<String, ResourceDescriptor> {
        BTreeMap::new()
    }

    // ---- Contract registry ------------------------------------------------

    #[test]
    fn contract_registry_rejects_duplicate_registration() {
        let mut c = ContractRegistry::new();
        c.register(
            "source.create@1".parse().unwrap(),
            Box::new(SourceStub(linear_premul())),
        )
        .unwrap();
        let err = c
            .register(
                "source.create@1".parse().unwrap(),
                Box::new(SourceStub(linear_premul())),
            )
            .unwrap_err();
        assert_eq!(err.class, ErrorClass::Schema);
        assert_eq!(err.code, E_DUPLICATE_CONTRACT);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn consistency_passes_when_manifest_and_contract_agree() {
        contracts(linear_premul())
            .check_consistency(&registry())
            .unwrap();
    }

    #[test]
    fn consistency_fails_when_contract_has_no_manifest() {
        let mut c = ContractRegistry::new();
        c.register(
            "source.create@1".parse().unwrap(),
            Box::new(SourceStub(linear_premul())),
        )
        .unwrap();
        // A registry missing source.create@1.
        let manifests = OperationRegistry::from_manifests([op(
            "filter.blur@1",
            &[("image", ResourceKind::Image)],
            &[("image", ResourceKind::Image)],
        )])
        .unwrap();
        let err = c.check_consistency(&manifests).unwrap_err();
        assert_eq!(err.code, E_CONTRACT_NOT_FOUND);
    }

    // ---- Happy path -------------------------------------------------------

    fn pipeline_plan() -> Plan {
        parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [
                    {"id": "src", "op": "source.create@1"},
                    {"id": "blur", "op": "filter.blur@1", "in": {"image": "node:src/image"}}
                ],
                "exports": {"final": {"resource": "node:blur/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn checks_a_linear_pipeline_and_infers_descriptors() {
        let plan = pipeline_plan();
        let reg = registry();
        let graph = resolve_plan(&plan, &reg).unwrap();
        let checked: CheckedGraph = check_graph(
            &plan,
            &graph,
            &reg,
            &contracts(linear_premul()),
            &no_inputs(),
        )
        .unwrap();

        assert_eq!(checked.output("src", "image"), Some(&linear_premul()));
        assert_eq!(checked.output("blur", "image"), Some(&linear_premul()));
        assert_eq!(checked.exports().len(), 1);
        assert_eq!(checked.exports()[0].0, "final");
        assert_eq!(checked.exports()[0].1, linear_premul());
    }

    // ---- Color / alpha mismatches (raised by the op contract) -------------

    #[test]
    fn srgb_into_a_linear_blur_is_a_semantic_color_error() {
        let plan = pipeline_plan();
        let reg = registry();
        let graph = resolve_plan(&plan, &reg).unwrap();
        let srgb = image(ColorEncoding::Srgb, AlphaRepresentation::Premultiplied);
        let err = check_graph(&plan, &graph, &reg, &contracts(srgb), &no_inputs()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Semantic);
        assert_eq!(err.code, "E_COLOR_ENCODING_MISMATCH");
        // The error is attributed to the consuming node.
        assert_eq!(err.context.node.as_deref(), Some("blur"));
    }

    #[test]
    fn straight_alpha_into_a_premultiplied_blur_is_an_alpha_error() {
        let plan = pipeline_plan();
        let reg = registry();
        let graph = resolve_plan(&plan, &reg).unwrap();
        let straight = image(ColorEncoding::LinearSrgb, AlphaRepresentation::Straight);
        let err = check_graph(&plan, &graph, &reg, &contracts(straight), &no_inputs()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Semantic);
        assert_eq!(err.code, "E_ALPHA_REPRESENTATION_MISMATCH");
        assert_eq!(err.context.node.as_deref(), Some("blur"));
    }

    // ---- Kind mismatch (raised by the checker's kind gate) ----------------

    #[test]
    fn feeding_a_mask_into_an_image_port_is_a_type_error() {
        // masked_replace wires `image` <- input:m (a Mask) -> kind gate fails.
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {"m": {"kind": "mask.file", "path": "m.png"}, "img": {"kind": "image.file", "path": "i.png"}},
                "nodes": [
                    {"id": "c", "op": "composite.masked_replace@1",
                     "in": {"image": "input:m", "mask": "input:img"}}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        let reg = registry();
        let graph = resolve_plan(&plan, &reg).unwrap();
        let mut inputs = BTreeMap::new();
        inputs.insert("m".to_owned(), mask());
        inputs.insert("img".to_owned(), linear_premul());

        let err =
            check_graph(&plan, &graph, &reg, &contracts(linear_premul()), &inputs).unwrap_err();
        assert_eq!(err.class, ErrorClass::Type);
        assert_eq!(err.code, E_PORT_KIND_MISMATCH);
        assert_eq!(err.context.node.as_deref(), Some("c"));
        assert_eq!(err.context.actual.as_deref(), Some("Mask"));
        assert_eq!(err.context.expected.as_deref(), Some("Image"));
    }

    #[test]
    fn well_typed_masked_replace_passes_the_kind_gate() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {"img": {"kind": "image.file", "path": "i.png"}, "m": {"kind": "mask.file", "path": "m.png"}},
                "nodes": [
                    {"id": "c", "op": "composite.masked_replace@1",
                     "in": {"image": "input:img", "mask": "input:m"}}
                ],
                "exports": {"out": {"resource": "node:c/image", "kind": "image", "path": "o.png"}}
            }"#,
        )
        .unwrap();
        let reg = registry();
        let graph = resolve_plan(&plan, &reg).unwrap();
        let mut inputs = BTreeMap::new();
        inputs.insert("img".to_owned(), linear_premul());
        inputs.insert("m".to_owned(), mask());

        let checked =
            check_graph(&plan, &graph, &reg, &contracts(linear_premul()), &inputs).unwrap();
        assert_eq!(checked.output("c", "image"), Some(&linear_premul()));
        assert_eq!(checked.exports()[0].1, linear_premul());
    }

    // ---- Missing input descriptor -----------------------------------------

    #[test]
    fn missing_input_descriptor_is_a_type_error() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {"src": {"kind": "image.file", "path": "i.png"}},
                "nodes": [
                    {"id": "blur", "op": "filter.blur@1", "in": {"image": "input:src"}}
                ],
                "exports": {}
            }"#,
        )
        .unwrap();
        let reg = registry();
        let graph = resolve_plan(&plan, &reg).unwrap();
        // `src` descriptor deliberately not supplied.
        let err = check_graph(
            &plan,
            &graph,
            &reg,
            &contracts(linear_premul()),
            &no_inputs(),
        )
        .unwrap_err();
        assert_eq!(err.class, ErrorClass::Type);
        assert_eq!(err.code, E_MISSING_INPUT_DESCRIPTOR);
        assert_eq!(err.context.node.as_deref(), Some("blur"));
    }

    // ---- Missing contract --------------------------------------------------

    #[test]
    fn a_node_without_a_registered_contract_fails() {
        let plan = parse_plan(
            r#"{
                "paintop": "1.0",
                "inputs": {},
                "nodes": [{"id": "src", "op": "source.create@1"}],
                "exports": {}
            }"#,
        )
        .unwrap();
        let reg = registry();
        let graph = resolve_plan(&plan, &reg).unwrap();
        // An empty contract registry: the contract is missing.
        let err =
            check_graph(&plan, &graph, &reg, &ContractRegistry::new(), &no_inputs()).unwrap_err();
        assert_eq!(err.class, ErrorClass::Reference);
        assert_eq!(err.code, E_CONTRACT_NOT_FOUND);
        assert_eq!(err.context.node.as_deref(), Some("src"));
    }
}
