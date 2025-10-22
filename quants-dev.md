# MXFP8 & NVFP4 Enablement For SM120 (Blackwell)

## Prerequisites & Baseline
- Require CUDA toolkit ≥ 12.8 (prefer 13.x for official SM120 support) and driver ≥ r560.
- Ensure cuBLASLt runtime ≥ 12.8.0; surface a hard error if the shared object is older even when headers compile.
- Confirm build targets include `sm_120`/`sm_120a` in `CMAKE_CUDA_ARCHITECTURES`, Python wheel metadata (`NVTE_CUDA_ARCHS`), and CI runners.
- Add release toggle (env var or build flag) so we can temporarily gate MXFP8/NVFP4 on SM120 while validating.

## 1. Runtime Capability Detection
1. **C++ backend (`transformer_engine/common/transformer_engine.cpp`)**
   - ✅ Replace the hard-coded arithmetic in `nvte_is_non_tn_fp8_gemm_supported()` with a probe that:
     - Queries current device SM (via `cuda::sm_arch`).
     - Checks runtime cuBLASLt version (`cublasLtGetVersion`) and CUDA runtime version.
     - Returns true for SM ≥ 100 when cuBLASLt supports MXFP8 all-layout GEMM (≥12.8), including SM120.
   - ✅ Introduce `nvte_is_nvfp4_supported()` (or equivalent) that mirrors the logic for NVFP4 block GEMM availability.
   - ✅ Export new probes through the C API header and bindings (PyTorch & JAX pybind modules).

2. **PyTorch front-end (`transformer_engine/pytorch/quantization.py`, `pytorch/csrc/quantizer.cpp`)**
   - ✅ Update `check_mxfp8_support()` and `check_nvfp4_support()` to rely on the new C++ probes instead of open-coded CC checks; include CUDA runtime / cuBLASLt version assertions and clear error messages.
   - Ensure `QuantizationManager` stops forcing BF16 fallback when the probes succeed.
   - Adjust `Float8Quantizer::create_tensor` logic so SM120 keeps row & column buffers consistent with true all-layout GEMM support.

3. **JAX bindings (`transformer_engine/jax/...`)**
   - Update `is_fp8_gemm_with_all_layouts_supported()` and any NVFP4 gating helpers to consume the new runtime probes.
   - Plumb capability flags through JAX pybind (`jax/csrc/extensions/pybind.cpp`) and expose Python helpers.

4. **Runtime config helpers**
   - Audit any helper that builds `runtime_config` (e.g., distributed initialisation) to stop pessimistically downgrading recipes when SM ≥ 120.

## 2. CUDA Datatype & Macro Updates
- Extend `transformer_engine/common/common.cu::get_cuda_dtype` to map `DType::kFloat8E8M0` and guard `DType::kFloat4E2M1` behind CUDA ≥ 12.8.
- In `common/util/ptx.cuh`, add SM120 feature macros so `float_to_e8m0`, NVFP4 transpose kernels, Hadamard transforms, and cp.async paths use the fast inline assembly on Blackwell (treat SM120 similarly to SM100/SM103).
- Confirm any CUTLASS templates currently specialised for `SM100` are callable for `SM120`; add aliases if CUTLASS expects dedicated tags.

-## 3. cuBLASLt GEMM Integration (`common/gemm/cublaslt_gemm.cu`)
- ✅ Register the new CUDA datatypes for MXFP8 (E8M0 scales) and NVFP4 (E2M1) in matrix layouts / describe objects.
- ✅ Update `CanonicalizeGemmInput` to:
  - ✅ Accept row/column data permutations for all TN/NT/NN/TT combinations when the SM120 probe passes.
  - ✅ Avoid redundant columnwise buffers when both layouts exist.
- ✅ Select the appropriate compute type for Blackwell (`CUBLAS_COMPUTE_32F_FAST_FP8XMMA` and FP4 equivalent) instead of generic FP32.
- ✅ Push cuBLASLt descriptor attributes for block scaling:
  - `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` for MXFP8,
  - `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` for NVFP4,
  - Set pointer/scale modes for the new datatype sizes.
- ✅ Revisit heuristic caching to include MXFP8/NVFP4 combos so autotuning does not reject new layouts.
- ✅ Mirror changes in the `nvte_cublas_gemm_v2` wrapper (alpha/beta device pointers, workspace alignment) to guarantee NVFP4 paths run without manual overrides.

## 4. Quantization Kernel Work
1. **Shared config (`common/common.h`, `QuantizationConfig`)**
   - ✅ Add explicit knobs for block length (16 vs 32) and scale format (E8M0/E4M3) so kernels don’t rely on hard-coded constants.
   - ✅ Ensure `QuantizationConfigWrapper` in PyTorch populates these knobs when constructing MXFP8 or NVFP4 recipes. (JAX pending)

2. **MXFP8 kernels (`common/util/cast_kernels.cuh`, `cast_gated_kernels.cuh`, `dequantize_kernels.cuh`)**
   - Generalise load/store loops to allow alternative block lengths where needed; keep warp scheduling correct for SM120’s wider shared-memory banks.
   - Use the SM120 inline assembly paths for E8M0 conversion; maintain portable fallback for older GPUs.
   - Confirm the TMA descriptors used for CP async copies encode the 32-element MXFP8 tiles correctly for SM120 caching behaviour.

3. **NVFP4 kernels (`common/recipe/nvfp4.cu`, `common/util/nvfp4_transpose.cuh`, `transpose/quantize_transpose_vector_blockwise_fp4.cu`)**
   - Enable the cp.async Bulk and TMEM intrinsics for SM120; add specialisations if CUTLASS/CUTE expects `SM120` tags.
   - Ensure the 2D quantisation path emits both the E4M3 block scales and the per-tensor FP32 guard, and that stochastic rounding / random Hadamard transforms can be toggled per config.
   - Verify the kernels respect the new block-size attribute instead of literal 16s.

## 5. Framework Integration
1. **PyTorch tensors & optimizers**
   - Extend `MXFP8TensorStorage` / `NVFP4TensorStorage` to carry the extra metadata (block size, guard scale) required by the updated kernels.
   - Upgrade weight master-copy helpers (`cast_master_weights_to_fp8`, grad accumulation buffers) to maintain MXFP8/NVFP4 mirrors automatically; ensure CUDA graph capture keeps metadata resident.
   - Remove BF16 fallback guards in TELinear, LayerNormLinear, LayerNormMLP, SSM/Hyena wrappers once capability probe succeeds.
   - Update distributed collectives (`pytorch/distributed.py`) so all-gather / reduce-scatter paths understand MXFP8 layouts and call the NVFP4 post-processing utilities.

2. **JAX quantisation stack** *(optional – to be handled separately)*
   - Teach `scaling_modes.py`, `quantizer.py`, and GEMM wrappers about the new block-size metadata and capability signals.
   - Ensure the FFI bridging in `jax/csrc/extensions/gemm.cpp` requests the right descriptors and that the Python recipe constructors expose MXFP8BlockScaling/NVFP4BlockScaling on SM120.

## 6. Build, Packaging, and CI
- Update wheel build scripts to ship CUDA 12.8+/13.x wheels containing SM120 cubins; refresh `setup.py` metadata.
- Adjust CI to compile MXFP8/NVFP4 tests for SM120, adding hardware coverage where possible (or provide emulation checks).
- Add feature flags to nightly packaging to avoid distributing partial SM120 support.

## 7. Testing & Validation
- Extend existing CUDA unit tests:
  - `tests/cpp/operator/test_cast_nvfp4_transpose.cu` and friends to run against SM120, validating both 1D & 2D quantisation modes.
  - Add GEMM smoke tests that exercise each transpose layout for MXFP8/NVFP4 with real cuBLASLt calls.
- Add PyTorch integration tests that perform forward/backward and optimizer steps in MXFP8 and NVFP4 modes on SM120.
- Include distributed tests (all-gather, reduce-scatter) to confirm scale metadata is exchanged correctly.
- For JAX, add reference comparisons vs FP16/BF16 pipelines.

## 8. Documentation & Messaging
- Update user guides, recipe docs, and release notes to describe:
  - Required CUDA/driver versions and supported hardware (SM120/Blackwell).
  - How to select the new recipes (`mxfp8`, `nvfp4`) end-to-end.
  - Any limitations (e.g., kernels still BF16-only).
- Provide migration guidance for users upgrading from Hopper (layout differences, new config flags).

## 9. Post-Enablement Validation
- Benchmark representative models comparing Hopper vs Blackwell to confirm parity / gains.
- Monitor runtime telemetry (if available) to ensure capability probes aren’t misfiring on unsupported drivers.
- Prepare rollback plan in case cuBLASLt regressions are found (build-time flag to disable MXFP8/NVFP4 on SM120).
