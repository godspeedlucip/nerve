### Task 5: Documentation, verification, and regression cleanup

**Files:**
- Modify: `TRAINING_INFERENCE_GUIDE.md` only where the new unified configuration/preset model needs explanation; preserve existing user edits.
- Modify: `verify_modules.py` to call the unified composition path where practical without removing existing proofs.
- Test: `nnunetv2/tests/unit/test_unified_pipeline_smoke.py`

**Interfaces:**
- Consumes all public interfaces from Tasks 1–4.
- Produces a documented migration path and a single CPU smoke test covering Aniso + Tubular + ProjCons + DTM active together.

- [ ] **Step 1: Write failing smoke test** that builds a tiny backbone and all-enabled pipeline, computes the composite loss with DTM auxiliary targets, performs backward, and asserts all enabled component losses are finite.
- [ ] **Step 2: Run the smoke test** with `pytest nnunetv2/tests/unit/test_unified_pipeline_smoke.py -q`; expected failure until all integration interfaces are wired.
- [ ] **Step 3: Implement** only the minimal documentation and verification-script changes required by the final public interfaces.
- [ ] **Step 4: Run the complete unit suite** with `pytest nnunetv2/tests/unit -q`; expected zero failures.
- [ ] **Step 5: Run static/import checks** for the touched modules and the existing module verification entrypoint where dependencies permit; record any environment-only limitations separately.
- [ ] **Step 6: Commit** with `test: verify unified enhancement pipeline`.
