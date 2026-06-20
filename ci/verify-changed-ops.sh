#!/usr/bin/env bash
# Run `cargo xtask verify-op` for each changed op manifest (AGENT_VERIFICATION
# §8.1 changed-op verification, §14). Invoked by the `changed-op` CI job with the
# newline-separated manifest paths in $CHANGED_MANIFESTS.
#
# For each manifest it reads the canonical op id from the file's `id` field and
# runs verify-op against it, writing the report tree under
# `target/verification/<op-id>/`. Any incomplete op makes verify-op exit
# non-zero, which fails this script (and the job).
set -euo pipefail

if [ -z "${CHANGED_MANIFESTS:-}" ]; then
  echo "verify-changed-ops: no changed manifests; nothing to do."
  exit 0
fi

# Resolve the xtask invocation. CI builds and runs it through cargo; the
# integration test sets $XTASK_BIN to a prebuilt binary to avoid nesting cargo.
if [ -n "${XTASK_BIN:-}" ]; then
  run_xtask() { "$XTASK_BIN" "$@"; }
else
  # Build xtask once up front so per-op runs are fast and a compile error fails
  # clearly rather than mid-loop.
  cargo build -p xtask --quiet
  run_xtask() { cargo run -p xtask --quiet -- "$@"; }
fi

status=0
while IFS= read -r manifest; do
  # Skip blank lines (the env var can carry a trailing newline).
  [ -n "$manifest" ] || continue
  if [ ! -f "$manifest" ]; then
    echo "verify-changed-ops: $manifest no longer exists; skipping."
    continue
  fi

  op=$(jq -r '.id' "$manifest")
  if [ -z "$op" ] || [ "$op" = "null" ]; then
    echo "verify-changed-ops: $manifest has no \`id\`; cannot verify." >&2
    status=1
    continue
  fi

  echo "::group::verify-op $op ($manifest)"
  if run_xtask verify-op "$op" --manifest "$manifest"; then
    echo "verify-op $op: PASS"
  else
    echo "verify-op $op: FAILED" >&2
    status=1
  fi
  echo "::endgroup::"
done <<EOF
${CHANGED_MANIFESTS}
EOF

exit "$status"
