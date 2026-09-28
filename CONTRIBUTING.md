# Contributing to Telemetry Resilience

## Development install

```bash
git clone <repo>
cd telemetry-resilience
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

## Running tests

```bash
pytest
```

The full test suite must pass before a change is merged. Run the full
suite; there is no separate fast path.

## Adding a new fault operator

1. Create a module under `src/telemetry_resilience/faults/` (e.g.
   `myfault.py`) implementing the operator with the same call signature and
   validation style as the existing operators. Reuse the helpers in
   `faults/_common.py` where they apply.
2. Register the operator in `src/telemetry_resilience/faults/__init__.py` and
   in the fault-type dispatch/validation used by scenario loading.
3. Add a row to the **Fault operators** table in `README.md`.
4. Add tests covering: basic behavior (manifest records actual affected
   windows/counts), determinism (same seed → same values), dtype safety
   (raising a `FaultError` with an explicit `cast:` hint instead of silently
   widening or nulling a dtype), and invalid parameters (bad window,
   unknown channel, out-of-range config → clear errors).
5. Document any special semantics (e.g. whether rows are kept or deleted,
   how the manifest reports it) in the README's **Fault semantics that
   matter** section if the operator introduces a distinction like
   `dropout` vs `timestamp_gap`.

## Adding a regression test

Bug fixes **must** include a regression test that fails without the fix and
passes with it. Put it in the appropriate existing test module (e.g.
`tests/test_run22_regressions.py` for suite/runner fixes), or create a new
module following the `test_run<N>_regressions.py` naming convention for the
release the fix lands in. The test should:

- reproduce the bug with a minimal scenario, input, and target,
- assert the corrected behavior (manifest content, exit code, artifact
  layout, or error message — whichever the fix is about),
- be deterministic: fixed seeds, no wall-clock assertions, no network.

## Style expectations

- **No silent behavior.** Prefer loud, specific errors over guessing
  (the cast rule in `README.md` is the model). Never silently widen a dtype,
  skip a fault, or execute something the user didn't request.
- **Manifests record ground truth.** Any operator or runner change must keep
  manifests reporting what actually happened, not what was requested.
- **Determinism.** New randomness must be seeded from the scenario/case
  seed; same inputs must produce identical corrupted values.
- **Docs in the same change.** Update `README.md`, `CHANGELOG.md` (under an
  "Unreleased" section), and any affected examples alongside the code.
- Keep the CLI surface minimal: new flags need a documented reason in the
  README.
