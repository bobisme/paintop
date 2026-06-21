# Performance baselines (M3 exit criterion 3)

This directory holds the **checked-in performance baseline artifacts** the M3
exit gate (`ci/m3-gate.sh`, criterion 3) requires (`plan.md` §19 M3). A baseline
records, per `(op, backend, size)`, the measured throughput in megapixels per
second — the size-normalized rate the regression check compares.

## How a baseline is produced

```
cargo run --release -p xtask -- perf-baseline \
    --out target/verification/perf/baseline.json \
    --machine <runner-id>
```

The `xtask perf-baseline` driver sweeps:

- the optimized-CPU pointwise kernels on both `cpu.reference` and
  `cpu.optimized` (a single timing yields both rows; the speedup is implicit in
  the throughput ratio), and
- the `wgpu.separable` two-pass Gaussian **when a GPU adapter is present** (GPU
  rows are skipped cleanly with no adapter — the artifact still records the CPU
  rows, exactly like the differential harness's GPU gating).

The emitted `baseline.json` is the artifact CI **uploads** every run.

## Why it is machine-tolerant (no absolute wall-clock gate)

Absolute throughput is meaningless across CI runners, so the gate never asserts
an absolute number. The regression check is purely **relative**: a row is flagged
only when its throughput drops below `baseline × (1 − threshold)` against a
reference captured on the **same machine class**. The default slack threshold is
generous (25%) and configurable via `--threshold`; a row with no matching
baseline records a new point rather than failing.

```
cargo run --release -p xtask -- perf-baseline \
    --out target/verification/perf/current.json \
    --baseline ci/perf/baseline.<machine>.json \
    --machine <machine> --threshold 0.25
```

A non-zero exit means at least one row regressed beyond the threshold.

## The checked-in references

- `baseline.rtx3090-dev.json` — captured on the M3 development host (NVIDIA
  RTX 3090, Vulkan, release build). It includes the `wgpu.separable` GPU rows.

A runner with no matching checked-in baseline for its machine class simply
records a fresh one (every row reports `no-baseline`, the report stays clean) —
that artifact becomes the reference the next run compares against. This keeps the
gate honest without committing hardware-specific numbers as a hard pass/fail bar.
