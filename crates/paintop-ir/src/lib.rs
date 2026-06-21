//! `paintop-ir`: canonical IR types, the plan/op-manifest schema, the central
//! error taxonomy, and canonicalization + hashing.
//!
//! This is the base crate of the workspace; nothing else in `paintop` may sit
//! below it (see `plan.md` §6.1). Concrete types are filled in by later bones.

pub mod canonical;
pub mod check;
pub mod contract;
pub mod error;
pub mod hash;
pub mod limits;
pub mod manifest;
pub mod normalize;
pub mod plan;
pub mod registry;
pub mod resolve;
pub mod resource;
pub mod scan;
pub mod verify;

pub use canonical::{E_NON_FINITE_FLOAT, to_canonical_bytes, to_canonical_string};
pub use check::{
    CheckedGraph, ContractRegistry, E_CONTRACT_NOT_FOUND, E_DUPLICATE_CONTRACT,
    E_MISSING_INPUT_DESCRIPTOR, E_OUTPUT_PORT_NOT_INFERRED, E_PORT_KIND_MISMATCH, check_graph,
};
pub use contract::{
    AssertionResult, AssertionStatus, ContractError, Descriptors, E_CONTRACT_PORT_MISMATCH,
    InputRegions, OpContract, OutputDescriptors, OutputRegions, check_contract_consistency,
};
pub use error::{Error, ErrorClass, ErrorContext, ErrorEnvelope, ErrorPayload, Result, Suggestion};
pub use hash::{
    BLAKE3_PREFIX, E_INVALID_HASH, HashDomain, SemanticHash, hash_canonical_bytes, hash_value,
};
pub use limits::{
    E_MAX_DEPTH, E_MAX_INLINE_PAYLOAD, E_MAX_NODES, E_MAX_PLAN_BYTES, PlanLimits, check_limits,
};
pub use manifest::{
    CPU_REFERENCE_BACKEND, CPU_REFERENCE_NAME, DeterminismTier, ImplId, InputSpec, OpId,
    OperationManifest, OutputSpec, ParamSpec, ParamType, ParamUnit, ResourceKind, RoiCategory,
    RoiPolicy, TestMetadata, manifest_json_schema,
};
pub use normalize::{normalize, normalized_value, semantic_hash};
pub use plan::{Extensions, InputDecl, Node, Plan, parse_plan};
pub use registry::{
    E_DUPLICATE_OP_ID, E_OP_NOT_FOUND, E_OP_VERSION_UNSUPPORTED, OperationRegistry,
};
pub use resolve::{
    E_DANGLING_REFERENCE, E_DUPLICATE_NODE_ID, E_GRAPH_CYCLE, E_INVALID_NODE_ID,
    E_INVALID_REFERENCE, E_MISSING_INPUT_PORT, E_UNKNOWN_INPUT_PORT, E_UNKNOWN_OUTPUT_PORT,
    Reference, ResolvedExport, ResolvedGraph, ResolvedNode, resolve_plan,
};
pub use resource::{
    AlphaRepresentation, AssertionOutcome, AssertionSeverity, BoundaryMode, ChannelLayout,
    ChannelStats, ColorEncoding, ColorRange, CoordinateConvention, DiffMetrics, Extent, FieldArity,
    FieldDescriptor, ImageDescriptor, MaskDescriptor, MaskMeaning, Rect, Report, ReportDescriptor,
    RequestedColorEncoding, ResourceDescriptor, ScalarType, SdfDescriptor, SdfSign, SdfUnits,
    SemanticRole, ValidRange, VectorEncoding, VectorNormalization, VectorSpace,
};
pub use scan::{E_DUPLICATE_KEY, E_INVALID_NUMBER, scan_json};
pub use verify::{
    CategoryStatus, E_VERIFY_CATEGORY_MISSING, E_VERIFY_CATEGORY_NOT_APPLICABLE,
    E_VERIFY_NA_REASON_MISSING, VerificationCategory, VerificationDeclarations, verify_categories,
};
