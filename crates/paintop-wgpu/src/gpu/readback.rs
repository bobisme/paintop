//! No-unplanned-readback verification for fully GPU-compatible chains (`plan.md`
//! §12.3, §19 M3 exit criterion 2; bn-3q0).
//!
//! A chain of GPU-compatible nodes (fused pointwise + separable filter + splat) is
//! meant to keep its intermediates **GPU-resident**: the only host↔GPU readbacks are
//! the *declared* materialization points — the final export, and any explicit
//! `debug.materialize` evidence barrier (`plan.md` §18). An *unplanned* readback (a
//! per-stage round trip the scheduler did not ask for) silently defeats the whole
//! point of a GPU backend and must be caught.
//!
//! This module surfaces readback **events** in the execution trace and verifies the
//! count against the plan's declared materialization points:
//!
//! * [`ReadbackEvent`] — one host↔GPU transfer, tagged with *why* it happened
//!   (an export, a `debug.materialize` barrier, or — the bug we reject — an
//!   unplanned intermediate round trip);
//! * [`ChainTrace`] — the ordered events a GPU chain produced, with a
//!   [`verify`](ChainTrace::verify) that asserts **zero** unplanned readbacks and
//!   exactly the declared number of planned ones;
//! * [`ChainStage`] — the per-node descriptor a chain is built from, so a trace is
//!   assembled from real stage outputs ([`ExecutionTrace`]) rather than hand-counted.
//!
//! The model is `wgpu`-free so the accounting is unit-testable GPU-less; a GPU
//! integration test (`tests/readback_trace_gpu.rs`) drives the *real* pointwise →
//! filter → splat chain and feeds each stage's [`ExecutionTrace`] in, proving the
//! live chain is readback-free except at the declared export, and that inserting a
//! host-side `debug.materialize` adds exactly one expected readback.

use super::fusion::MATERIALIZE_OP;
use super::pointwise::ExecutionTrace;

/// Why a host↔GPU readback happened, for the no-unplanned-readback assertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadbackReason {
    /// The plan's declared final export/materialization of the chain result. This is
    /// the *one* readback a fully-GPU-resident chain is allowed at its end.
    Export,
    /// A host-side `debug.materialize@1` evidence barrier (`plan.md` §18): a
    /// deliberately-planned readback that forces the intermediate to exist as a real
    /// buffer for evidence. Each barrier is exactly one expected readback.
    Materialize,
    /// An **unplanned** intermediate readback — a per-stage host round trip the
    /// scheduler did not ask for. This is the defect [`ChainTrace::verify`] rejects:
    /// a GPU-compatible chain must keep its intermediates on the device.
    UnplannedIntermediate,
}

impl ReadbackReason {
    /// Whether this readback was *planned* (an export or a `debug.materialize`).
    #[must_use]
    pub const fn is_planned(self) -> bool {
        matches!(self, Self::Export | Self::Materialize)
    }
}

/// One readback event surfaced in a chain's execution trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadbackEvent {
    /// The 0-based index of the chain stage that produced the readback.
    pub stage: usize,
    /// Why the readback happened.
    pub reason: ReadbackReason,
}

/// One node of a GPU-compatible chain, as the readback model sees it.
///
/// A chain is an ordered list of these. A [`GpuKernel`](Self::GpuKernel) stage carries
/// the real per-op [`ExecutionTrace`] its `run_*` produced; a
/// [`Materialize`](Self::Materialize) stage is a host-side `debug.materialize@1`
/// barrier that forces a readback. The model walks the chain and decides which of a
/// stage's readbacks are planned (the chain's final export, or a barrier) versus
/// unplanned (a per-stage round trip in the *middle* of the chain).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainStage {
    /// A GPU kernel stage (fused pointwise / separable filter / splat) and the trace
    /// its `run_*` reported.
    GpuKernel(ExecutionTrace),
    /// A host-side `debug.materialize@1` evidence barrier: forces exactly one
    /// readback of the running intermediate (`plan.md` §18).
    Materialize,
}

impl ChainStage {
    /// The op id a `Materialize` stage corresponds to (`debug.materialize@1`).
    #[must_use]
    pub const fn materialize_op() -> &'static str {
        MATERIALIZE_OP
    }
}

/// The ordered readback events a GPU chain produced, for the no-unplanned-readback
/// assertion.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChainTrace {
    events: Vec<ReadbackEvent>,
}

impl ChainTrace {
    /// Build the readback trace for an ordered chain of stages.
    ///
    /// The rule (`plan.md` §13.2, §19 M3): a fully-GPU-resident chain reads back only
    /// at *declared* points. So:
    ///
    /// * a [`Materialize`](ChainStage::Materialize) stage always contributes **one**
    ///   [`Materialize`](ReadbackReason::Materialize) readback (the planned barrier);
    /// * the **last** stage's export readback is the chain's single
    ///   [`Export`](ReadbackReason::Export);
    /// * any export readback reported by a **non-last** GPU stage is an
    ///   [`UnplannedIntermediate`](ReadbackReason::UnplannedIntermediate) — the stage
    ///   should have left its output GPU-resident for the next stage, not round-tripped
    ///   it to the host;
    /// * any stage's `intermediate_readbacks > 0` is always unplanned.
    ///
    /// A GPU stage that *follows* a materialize barrier re-uploads from the
    /// materialized host buffer; that upload is not a readback and is not counted here
    /// (readbacks are GPU→host).
    #[must_use]
    pub fn build(stages: &[ChainStage]) -> Self {
        let mut events = Vec::new();
        let last_kernel = last_gpu_kernel_index(stages);
        for (i, stage) in stages.iter().enumerate() {
            match stage {
                ChainStage::Materialize => {
                    events.push(ReadbackEvent {
                        stage: i,
                        reason: ReadbackReason::Materialize,
                    });
                }
                ChainStage::GpuKernel(trace) => {
                    // Any per-stage intermediate readback is unplanned, always.
                    for _ in 0..trace.intermediate_readbacks {
                        events.push(ReadbackEvent {
                            stage: i,
                            reason: ReadbackReason::UnplannedIntermediate,
                        });
                    }
                    // An export readback is the planned final one *only* on the last
                    // stage; anywhere earlier it is an unplanned round trip.
                    let is_last_kernel = i == last_kernel;
                    for _ in 0..trace.export_readbacks {
                        events.push(ReadbackEvent {
                            stage: i,
                            reason: if is_last_kernel {
                                ReadbackReason::Export
                            } else {
                                ReadbackReason::UnplannedIntermediate
                            },
                        });
                    }
                }
            }
        }
        Self { events }
    }

    /// The recorded readback events, in chain order.
    #[must_use]
    pub fn events(&self) -> &[ReadbackEvent] {
        &self.events
    }

    /// The number of unplanned (intermediate round-trip) readbacks — the defect count.
    #[must_use]
    pub fn unplanned(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.reason == ReadbackReason::UnplannedIntermediate)
            .count()
    }

    /// The number of planned readbacks (exports + materialize barriers).
    #[must_use]
    pub fn planned(&self) -> usize {
        self.events.iter().filter(|e| e.reason.is_planned()).count()
    }

    /// The number of `debug.materialize` barrier readbacks in the chain.
    #[must_use]
    pub fn materialize_readbacks(&self) -> usize {
        self.events
            .iter()
            .filter(|e| e.reason == ReadbackReason::Materialize)
            .count()
    }

    /// Verify the chain incurred **no unplanned readback** and exactly
    /// `expected_planned` planned ones (`plan.md` §19 M3 exit criterion 2).
    ///
    /// A fully-GPU-resident pointwise+filter+splat chain expects `1` (the export);
    /// inserting one `debug.materialize` makes it `2` (export + one barrier).
    ///
    /// # Errors
    /// [`ReadbackViolation`] if any unplanned readback occurred, or the planned count
    /// differs from `expected_planned`.
    pub fn verify(&self, expected_planned: usize) -> Result<(), ReadbackViolation> {
        let unplanned = self.unplanned();
        if unplanned > 0 {
            return Err(ReadbackViolation::Unplanned {
                count: unplanned,
                events: self.events.clone(),
            });
        }
        let planned = self.planned();
        if planned != expected_planned {
            return Err(ReadbackViolation::PlannedMismatch {
                expected: expected_planned,
                actual: planned,
            });
        }
        Ok(())
    }
}

/// The index of the last GPU-kernel stage in a chain (the one whose export is the
/// chain's declared final readback). A `Materialize`-terminated chain still exports
/// from its last GPU kernel.
fn last_gpu_kernel_index(stages: &[ChainStage]) -> usize {
    stages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(i, s)| matches!(s, ChainStage::GpuKernel(_)).then_some(i))
        .unwrap_or(usize::MAX)
}

/// A readback-policy violation surfaced by [`ChainTrace::verify`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReadbackViolation {
    /// The chain incurred an unplanned intermediate readback — a per-stage host round
    /// trip the scheduler did not ask for.
    #[error("{count} unplanned intermediate readback(s) in a GPU-compatible chain: {events:?}")]
    Unplanned {
        /// How many unplanned readbacks occurred.
        count: usize,
        /// The full event list, for triage.
        events: Vec<ReadbackEvent>,
    },
    /// The chain's planned-readback count did not match the plan's declared
    /// materialization points.
    #[error("expected {expected} planned readback(s) but the chain performed {actual}")]
    PlannedMismatch {
        /// The expected number of planned readbacks (declared materialization points).
        expected: usize,
        /// The actual number observed.
        actual: usize,
    },
}

/// A GPU-resident kernel stage from its `run_*` [`ExecutionTrace`].
///
/// Convenience constructor so a chain is assembled from real stage outputs.
#[must_use]
pub const fn gpu_stage(trace: ExecutionTrace) -> ChainStage {
    ChainStage::GpuKernel(trace)
}

#[cfg(test)]
mod tests {
    use super::{ChainStage, ChainTrace, ReadbackReason, ReadbackViolation, gpu_stage};
    use crate::gpu::pointwise::ExecutionTrace;

    /// A GPU-resident *intermediate* kernel trace: uploaded once (or fed from the
    /// prior stage's resident buffer), no readbacks at all — its output stays on the
    /// device for the next stage. This is what a composed GPU-resident executor
    /// produces for a non-terminal stage.
    const fn resident(stages: u32) -> ExecutionTrace {
        ExecutionTrace {
            uploads: 1,
            intermediate_readbacks: 0,
            export_readbacks: 0,
            stages,
        }
    }

    /// The terminal GPU-resident kernel trace: one export readback (the declared
    /// final materialization), no intermediate readbacks.
    const fn terminal(stages: u32) -> ExecutionTrace {
        ExecutionTrace {
            uploads: 1,
            intermediate_readbacks: 0,
            export_readbacks: 1,
            stages,
        }
    }

    #[test]
    fn fully_resident_chain_has_one_export_and_no_unplanned_readback() {
        // pointwise -> filter -> splat, all GPU-resident: only the final export reads
        // back, zero intermediates.
        let chain = [
            gpu_stage(resident(4)),
            gpu_stage(resident(2)),
            gpu_stage(terminal(1)),
        ];
        let trace = ChainTrace::build(&chain);
        assert_eq!(trace.unplanned(), 0, "no intermediate round trips");
        assert_eq!(trace.planned(), 1, "exactly one declared export");
        assert_eq!(trace.materialize_readbacks(), 0);
        trace.verify(1).expect("readback-free except the export");
    }

    #[test]
    fn debug_materialize_introduces_exactly_one_expected_readback() {
        // Inserting a host-side debug.materialize between the filter and the splat
        // adds exactly one planned readback (the barrier), still zero unplanned.
        let chain = [
            gpu_stage(resident(4)),
            ChainStage::Materialize,
            gpu_stage(terminal(1)),
        ];
        let trace = ChainTrace::build(&chain);
        assert_eq!(trace.unplanned(), 0);
        assert_eq!(trace.materialize_readbacks(), 1, "one barrier readback");
        assert_eq!(trace.planned(), 2, "export + one materialize barrier");
        trace.verify(2).expect("export + one barrier");
        // And the strict no-barrier expectation now fails (the count changed).
        assert!(matches!(
            trace.verify(1),
            Err(ReadbackViolation::PlannedMismatch {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn a_mid_chain_export_is_an_unplanned_readback() {
        // A middle stage that reads its output back to the host (an export readback
        // anywhere but the end) is exactly the defect the verifier rejects: it should
        // have left its output GPU-resident for the next stage.
        let mid_export_chain = [
            gpu_stage(terminal(4)), // export mid-chain -> unplanned
            gpu_stage(resident(2)),
            gpu_stage(terminal(1)),
        ];
        let trace = ChainTrace::build(&mid_export_chain);
        assert!(trace.unplanned() >= 1, "a mid-chain export is unplanned");
        let err = trace.verify(1).expect_err("the leak must be rejected");
        assert!(matches!(err, ReadbackViolation::Unplanned { .. }));

        // A stage that ALSO did an explicit intermediate readback is caught too.
        let leaky = ExecutionTrace {
            uploads: 1,
            intermediate_readbacks: 1,
            export_readbacks: 0,
            stages: 2,
        };
        let leaky_chain = [
            gpu_stage(resident(4)),
            gpu_stage(leaky),
            gpu_stage(terminal(1)),
        ];
        let trace = ChainTrace::build(&leaky_chain);
        assert!(trace.unplanned() >= 1, "a mid-chain readback is unplanned");
        assert!(matches!(
            trace.verify(1),
            Err(ReadbackViolation::Unplanned { .. })
        ));
    }

    #[test]
    fn materialize_stage_names_the_evidence_barrier_op() {
        assert_eq!(ChainStage::materialize_op(), "debug.materialize@1");
    }

    #[test]
    fn empty_chain_is_vacuously_readback_free() {
        let trace = ChainTrace::build(&[]);
        assert_eq!(trace.unplanned(), 0);
        assert_eq!(trace.planned(), 0);
        trace.verify(0).expect("nothing to read back");
    }

    #[test]
    fn reason_planned_classification() {
        assert!(ReadbackReason::Export.is_planned());
        assert!(ReadbackReason::Materialize.is_planned());
        assert!(!ReadbackReason::UnplannedIntermediate.is_planned());
    }
}
