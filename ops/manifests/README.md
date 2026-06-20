# Operation manifests

Checked-in operation manifests (`plan.md` §6, `IR_SPEC` §3.3/§4/§6). Each file is
one operation version's declared contract: id, determinism tier, ROI policy,
typed ports, parameters, implementations, and verification declarations.

## Naming

One manifest per file, named for its canonical op id:

```text
ops/manifests/<namespace>.<name>@<major>.json   e.g. color.convert@1.json
```

The `@` and `.` in op ids are filesystem-safe, so the file name mirrors the
manifest's `id` field. The `verify-op` runner reads the id *from the manifest*,
so the report path matches the canonical id regardless of the file name.

## Changed-op CI

The `changed-op` job in `.github/workflows/ci.yml` runs `ci/verify-changed-ops.sh`
for every manifest under this directory that a pull request touches. The driver
runs `cargo xtask verify-op <id> --manifest <file>` and the job uploads the
resulting `target/verification/<op-id>/` report tree as a build artifact
(`AGENT_VERIFICATION` §8.1 "changed-op verification", §14). An incomplete op
(an applicable verification category that is neither covered nor
not-applicable-with-a-reason) makes `verify-op` exit non-zero and fails the job.

Real op manifests land with their op bones in segment 2.
