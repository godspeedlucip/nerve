### Task 2: Unified model enhancement pipeline

**Files:**
- Create: `nnunetv2/network_modules/enhancement_pipeline.py`
- Modify: `nnunetv2/network_modules/aniso_stent_unet.py`
- Modify: `nnunetv2/network_modules/__init__.py`
- Test: `nnunetv2/tests/unit/test_enhancement_pipeline.py`

**Interfaces:**
- Consumes `EnhancementConfig` from Task 1.
- Produces `EnhancementPipeline(backbone, in_channels, num_classes, config)` with delegated `decoder` and stable module attributes `aniso_attn`, `backbone`, `tub_enh`.
- `forward(x)` returns a tensor or list matching the backbone output type; TubularEnhancement only transforms index 0 of a deep-supervision list.

- [ ] **Step 1: Write failing tests** for all-off identity behavior, Aniso-only input transformation, Tubular-only final-output transformation, simultaneous Aniso+Tubular composition, deep supervision auxiliary-output preservation, decoder delegation, and stable state-dict keys.
- [ ] **Step 2: Run focused tests** with `pytest nnunetv2/tests/unit/test_enhancement_pipeline.py -q`; expected failure because `EnhancementPipeline` does not exist.
- [ ] **Step 3: Implement** the pipeline using existing `AnisotropicAttention` and `TubularEnhancement`; instantiate disabled modules as identity-safe paths while retaining stable compatibility attribute names.
- [ ] **Step 4: Adapt `AnisoStentUNet`** to delegate to or share the pipeline implementation without changing its constructor or state-dict names.
- [ ] **Step 5: Run focused tests** and the existing `verify_modules.py` CPU-compatible portion; expected all pipeline tests pass and existing module imports remain valid.
- [ ] **Step 6: Commit** with `refactor: unify enhancement model pipeline`.
