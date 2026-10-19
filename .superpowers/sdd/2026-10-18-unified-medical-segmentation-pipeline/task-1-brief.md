### Task 1: Configuration and auxiliary target contracts

**Files:**
- Create: `nnunetv2/training/pipeline_config.py`
- Create: `nnunetv2/training/auxiliary_targets.py`
- Modify: `nnunetv2/training/__init__.py`
- Test: `nnunetv2/tests/unit/test_pipeline_contracts.py`

**Interfaces:**
- Produces `EnhancementConfig`, `AuxiliaryTargets`, and `AuxiliaryTargets.empty()` for Tasks 2–4.
- `EnhancementConfig` exposes validated fields named in the spec and a `from_legacy_preset(name: str)` factory for compatibility presets.
- `AuxiliaryTargets.to(device)` and `AuxiliaryTargets.final_resolution()` preserve tensor/list shape semantics.

- [ ] **Step 1: Write failing tests** for default/legacy preset values, invalid negative weights, invalid DTM modes, recursive device movement, empty auxiliary targets, and mismatched batch/shape validation.
- [ ] **Step 2: Run the focused tests** with `pytest nnunetv2/tests/unit/test_pipeline_contracts.py -q`; expected failure because the new contracts do not exist.
- [ ] **Step 3: Implement** the frozen-style configuration dataclass and auxiliary target dataclass with explicit validation and shape errors containing actual shapes.
- [ ] **Step 4: Run the focused tests** again; expected all contract tests pass on CPU.
- [ ] **Step 5: Commit** with `feat: add unified pipeline contracts`.
