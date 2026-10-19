# Task 1 implementation report

## Files changed

- Created `nnunetv2/training/pipeline_config.py` with frozen `EnhancementConfig`, validation, and legacy trainer presets.
- Created `nnunetv2/training/auxiliary_targets.py` with frozen `AuxiliaryTargets`, recursive device movement, final-resolution selection, and shape validation.
- Updated `nnunetv2/training/__init__.py` to export both contracts.
- Created `nnunetv2/tests/unit/test_pipeline_contracts.py` with 17 focused contract tests.
- Preserved the pre-existing user modifications in `.gitignore` and `TRAINING_INFERENCE_GUIDE.md`; neither file was staged or changed by this task.

## RED

Command:

```text
python -m pytest nnunetv2/tests/unit/test_pipeline_contracts.py -q
```

Observed output:

```text
ImportError while importing test module ...
ModuleNotFoundError: No module named 'nnunetv2.training.auxiliary_targets'
1 error during collection
```

The initial `pytest ...` command could not be launched because `pytest` is not on PATH; the `python -m pytest` invocation above produced the expected missing-contract RED result.

## GREEN

Command:

```text
python -m pytest nnunetv2/tests/unit/test_pipeline_contracts.py -q
```

Observed output:

```text
.................                                                        [100%]
17 passed, 1 warning in 1.25s
```

## Final suite

Command:

```text
python -m pytest nnunetv2/tests -q
```

Observed output:

```text
.................                                                        [100%]
17 passed, 1 warning in 1.24s
```

## Commit

- Message: `feat: add unified pipeline contracts`
- Hash: `ce1606b`

## Concerns

- Pytest emits a deprecation warning from the installed `pytest-asyncio` plugin under Python 3.14 (`asyncio.get_event_loop_policy`); no test failed.
- The repository test tree currently collects only the 17 focused tests; integration scripts are present but are not pytest test modules.
- The report is intentionally written after the implementation commit and is not included in that commit, so the recorded hash remains the exact implementation commit hash.

## Fix report: contract validation hardening

### Fixes applied

- List-length mismatch errors now include the actual per-level DTM and `valid_mask` tensor shapes.
- Each non-`None` auxiliary tree is validated at construction, including nested structure, non-empty lists, and tensor-only leaves; paired trees are then checked for matching shapes.
- `EnhancementConfig` now validates boolean flags before use, requires a real integer epoch, and rejects non-finite numeric angles and weights.
- Legacy preset tests now assert compatibility-critical `dtm_weight` values: `0.0` for monitor mode and `1.0` for active GSL mode.

### Fix RED

Command:

```text
python -m pytest nnunetv2/tests/unit/test_pipeline_contracts.py -q
```

Exact result summary:

```text
............FFFFFFFFF.....FFFFF                                          [100%]
14 failed, 17 passed, 1 warning in 1.26s
```

The failures were the new regression assertions for non-finite values, non-integer/non-boolean types, per-level shape reporting, and single-field tree validation.

### Fix GREEN

Focused command:

```text
python -m pytest nnunetv2/tests/unit/test_pipeline_contracts.py -q
```

Exact result:

```text
...............................                                          [100%]
31 passed, 1 warning in 1.16s
```

Full available suite command:

```text
python -m pytest nnunetv2/tests -q
```

Exact result:

```text
...............................                                          [100%]
31 passed, 1 warning in 1.14s
```

### Fix commit

- Message: `fix: harden unified pipeline contracts`
- Hash: `9e92e71`

### Fix concerns

- The same existing `pytest-asyncio` Python 3.14 deprecation warning remains; no tests failed.
- Existing user modifications in `.gitignore` and `TRAINING_INFERENCE_GUIDE.md` remain preserved and unstaged.
