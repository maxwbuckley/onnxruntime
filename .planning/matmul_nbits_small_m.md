# MatMulNBits CUDA — small-M kernel optimization

**Branch:** `coreml-quickgelu` (will likely fork a new branch when implementation starts)
**Hardware tested:** RTX 5090 (Blackwell, sm_120) on WSL2, CUDA 12.4 toolkit, ORT 1.23.2 wheel
**Goal:** close the M ∈ {2..8} performance hole in the CUDA `MatMulNBits` operator.

---

## TL;DR

The CUDA `MatMulNBits` op has a specialized fast GEMV kernel at M=1 and a CUTLASS GEMM kernel for larger M, both via TRT-LLM's `fpA_intB_gemm`. **Between them — at M=2..16 — there is a cliff**: the GEMM kernel pads M up to its tile size (≈128) and reads weights 6–10× more than necessary, wasting bandwidth. This shows up immediately on speculative decoding, batched decode, and any workload that can't reduce to M=1.

We will plug the gap with a small-M extension of the existing in-tree `MatMulFloatInt4Kernel` and route M ∈ {2..8} to it before the CUTLASS path.

---

## Baseline measurements

Harness: `onnxruntime/test/python/quantization/bench_matmul_nbits_micro.py`
- IOBinding with pre-allocated GPU tensors, CUDA event timing.
- `inner_reps=200` to amortize launch overhead; `repeats=30`, `warmup=50`.
- **Min** is the headline metric (WSL2 + display GPU = clock jitter contaminates median/mean).
- Reports min, p10, median, mean, p90, max, stdev, TFLOPS@min, GB/s@min.
- Predicts dispatch path (in-tree GEMV / fpA_intB CUTLASS / dequant+cuBLAS fallback).

Full baseline CSV: `/tmp/bench_baseline_wheel.csv` (160 configs).

### M=1 → M=2 latency cliff (block_size=32, fp16, 4-bit, all on fpA_intB CUTLASS)

| Shape (N, K) | Layer (model) | M=1 | M=2 | Slowdown |
|---|---|---|---|---|
| 3072, 3072 | o_proj (Phi-3) | 24us | 38us | 1.58× |
| 4096, 4096 | o_proj (Llama-3-8B) | 22us | 49us | 2.24× |
| 9216, 3072 | qkv_proj (Phi-3) | 26us | 55us | 2.16× |
| 6144, 4096 | qkv_proj (Llama-3-8B) | 23us | 51us | 2.21× |
| 3072, 8192 | down_proj (Phi-3) | 30us | 57us | 1.90× |
| 4096, 14336 | down_proj (Llama-3-8B) | 27us | **171us** | **6.30×** |
| 16384, 3072 | gate_up (Phi-3) | 25us | **134us** | **5.33×** |
| 28672, 4096 | gate_up (Llama-3-8B) | 37us | **372us** | **10.02×** |
| 32064, 3072 | lm_head (Phi-3) | 43us | **300us** | **6.98×** |
| 128256, 4096 | lm_head (Llama-3-8B) | 219us | **1635us** | **7.47×** |

**Key tell**: for lm_head, M=2 (1635us) ≈ M=128 (1685us). CUTLASS is M-tile padding to ~128 and doing the same memory traffic regardless of how few rows you actually wanted.

### Bandwidth utilization (peak HBM ≈ 1.8 TB/s)

- M=1: 19% (small-N shapes) → 88% (gate_up 28672×4096) of peak
- M=2..16: 8–14% across the board (collapse)
- M=128–512: 4–8% (now compute-bound, see TFLOPS row)

### TFLOPS at large M (peak fp16-with-fp32-accum on 5090 ≈ 210 dense)

- M=128: 49–83 TFLOPS (23–40%)
- M=512: 112–154 TFLOPS (53–73%) — CUTLASS is fine here

---

## Code paths in `MatMulNBits` (CUDA)

Located in `onnxruntime/contrib_ops/cuda/quantization/`:

1. **fpA_intB CUTLASS** (`matmul_nbits.cc`, `USE_FPA_INTB_GEMM=ON`) — vendored TRT-LLM kernels. Has *two* sub-paths picked by the gemm profiler at construction time:
   - `fpA_intB_gemv` for very small M (the M=1 fast path)
   - `CutlassFpAIntBGemmRunner` for M>1 (the slow-at-small-M GEMM)
2. **In-tree GEMV** (`matmul_4bits.cu` `MatMulFloatInt4Kernel`) — pure FMA + shuffle reduction with PTX `lop3` for fast int4→fp16. Hard-gated to **M=1** at line 340: `if (n % kColsPerThreadBlock != 0 || k % 8 != 0 || m > 1) return false;`. Currently *bypassed entirely* when fpA_intB is on.
3. **Dequant + cuBLAS fallback** — only when fpA_intB is off (or for `reorder_idx` / T-typed zero_points). Not the bottleneck on stock wheels.

---

## Plan

### Phase 1 — done
- [x] 1a. Micro-benchmark harness (`bench_matmul_nbits_micro.py`).
- [x] 1b. Baseline sweep on pip wheel; identified M=2..8 cliff.
- [ ] 1c. Build ORT from source on this branch with CUDA. Recommended toolkit upgrade: CUDA 12.9 for native sm_120 (RTX 5090) codegen; not strictly required for the small-M kernel work since it targets bandwidth, not Blackwell-specific instructions.

### Phase 2 — target chosen
**Extend `MatMulFloatInt4Kernel` to handle M ∈ {1..8}**, then route small-M cases to it ahead of the CUTLASS path.

Why this and not the CUTLASS GEMM optimization or a Marlin-style kernel:
- The existing in-tree kernel is already structurally correct for the memory-bound regime.
- Loading each B-byte once and fanning across M outputs is a tiny edit (~50 lines).
- No CUTLASS surgery, no new dependencies, no SM_90/SM_120-specific code.
- Tractable in a single session; verifiable in isolation.

### Phase 3 — kernel implementation
**Concrete changes:**

1. **`matmul_4bits.cu`** — extend the kernel:
   - Add an `int kRowsPerBlock` template parameter (1, 2, 4, 8). Keep block dims as `(kWarpSize, kColsPerThreadBlock)`.
   - Each thread now maintains `kRowsPerBlock × 8` partial sums instead of `8`.
   - The K-reduction inner loop reads the same `uint32_t` of packed weights once, then loads `kRowsPerBlock` `uint4`s of activations (one per row) and accumulates into the row-specific partial sums.
   - Warp reduce produces `kRowsPerBlock × kColsPerThreadBlock` outputs per block.
   - Output indexing: `output[m_id * n + n_id]` becomes `output[(m_base + r) * n + n_id]` for r in `[0, kRowsPerBlock)`.
   - The `bf16` and `fp32` overloads of `AccumulateEightElements4b` need the same row-fan-out treatment.

2. **`matmul_4bits.cu` (TryMatMul4Bits)** — relax the gate:
   - Drop `m > 1`. Add `m <= kMaxRowsPerBlock` (say 8).
   - Pick `kRowsPerBlock` = nextPow2(m) clamped to {1, 2, 4, 8}, dispatch to the right template instantiation.
   - Adjust `blocks.y` if we want one block to cover multiple M-rows.
   - Recompute shared mem size — the scale buffer doesn't change since it's per-N, not per-M.

3. **`matmul_nbits.cc`** — re-route small-M:
   - Inside `ComputeInternal`, **before** the `has_fpA_intB_gemm_` branch (line 321), add an early call to `TryMatMulNBits` when `m <= 8 && m >= 1 && reorder_idx_data == nullptr && !zero_points->IsDataType<T>()`. If it returns true, return early.
   - This keeps the M=1 fast path (where the in-tree kernel already wins) and adds M=2..8 coverage.
   - For `m > 8`, fall through to the existing fpA_intB path unchanged.

4. **Correctness verification:**
   - Add to `bench_matmul_nbits_micro.py` a `--verify` flag that builds an fp32 reference (dequantize → matmul) and compares fp16 output max-abs and max-rel error.
   - Pass criterion: max abs ≤ 2e-3, max rel ≤ 5e-3 vs the fp32 reference. Tighter than the existing CUTLASS path; if we can't hit that we'll inspect.
   - Sanity test: M=1 should be bit-identical between old kernel and new kernel (no row fan-out).

### Phase 4 — end-to-end Phi-3-mini benchmark
- After phase 3 passes verification and shows the predicted micro speedup.
- Use `genai` or onnxruntime-genai to run quantized Phi-3-mini-4k-instruct end-to-end.
- Measure prefill latency (M=seq_len) and decode latency (M=1 today, M=k for spec decode if we wire that up).
- Report median across n≥30, batch sweep {1, 4, 8}, prompt-len sweep {32, 128, 512, 2048}.
- **In any PR description, lead with the end-to-end median number, not the kernel speedup.** The kernel change is large; the end-to-end impact will be smaller because attention/norm/rotary still take time. Don't conflate.

---

## Predicted impact (memory-bound model, 1.8 TB/s peak)

Kernel-level, M=2, block_size=32, fp16:

| Shape | Current | Predicted | Speedup |
|---|---|---|---|
| 28672×4096 (gate_up Llama-8B) | 372us | ~37us | **10×** |
| 128256×4096 (lm_head Llama-8B) | 1635us | ~220us | **7.4×** |
| 32064×3072 (lm_head Phi-3) | 300us | ~43us | **7×** |
| 4096×14336 (down Llama-8B) | 171us | ~27us | **6×** |
| 16384×3072 (gate_up Phi-3) | 134us | ~25us | **5.3×** |
| 9216×3072 (qkv Phi-3) | 55us | ~26us | **2.1×** |
| 4096×4096 (o_proj Llama-8B) | 49us | ~22us | **2.2×** |
| 3072×3072 (o_proj Phi-3) | 38us | ~24us | **1.6×** |

End-to-end on Phi-3-mini decode @ M=1: probably negligible (M=1 already fast).
End-to-end on Phi-3-mini decode @ M=4 (spec): potentially substantial. Will measure.

---

## Where things live

- Bench harness: `onnxruntime/test/python/quantization/bench_matmul_nbits_micro.py`
- Baseline CSV: `/tmp/bench_baseline_wheel.csv`
- Baseline log: `/tmp/bench_baseline_wheel.log`
- Kernel to modify: `onnxruntime/contrib_ops/cuda/quantization/matmul_4bits.cu` (also `matmul_8bits.cu` for 8-bit)
- Dispatch to modify: `onnxruntime/contrib_ops/cuda/quantization/matmul_nbits.cc`
- Header: `onnxruntime/contrib_ops/cuda/quantization/matmul_nbits.cuh`

## Open questions to resolve at resume

1. Build with `USE_FPA_INTB_GEMM=ON` or `OFF` for the dev cycle? Recommend ON, so we test against the same wheel users actually have. Add a separate OFF run for completeness.
2. CUDA toolkit upgrade to 12.9 — do it now (better SASS for sm_120, clean PTX-JIT noise) or defer (the small-M kernel is bandwidth-bound and unlikely to differ)? Recommend defer; revisit if the kernel itself isn't memory-bound at M=8.
3. Do we also extend `matmul_8bits.cu` symmetrically? Recommend yes — same change, same wins for 8-bit quantized models. Small extra work.
4. New branch name? Suggest `cuda-matmul-nbits-small-m`.

## Resume checklist (for tomorrow)

- [ ] Confirm plan still looks right; adjust phase 2 target if anything has changed.
- [ ] Decide on CUDA toolkit upgrade.
- [ ] Cut a new branch off `coreml-quickgelu` or `main` (verify no merge conflicts).
- [ ] Build ORT from source: `./build.sh --config Release --build_shared_lib --parallel --use_cuda --cuda_home /usr/local/cuda-12.4 --cudnn_home /usr/lib/x86_64-linux-gnu --build_wheel --skip_tests` (adjust flags).
- [ ] Re-run baseline against the source build to confirm same numbers as the wheel (sanity).
- [ ] Implement the kernel changes (phase 3).
- [ ] Verify correctness vs fp32 reference.
- [ ] Re-run micro-bench, A/B against baseline; require ≥3× kernel speedup at M=2 on the bad shapes before declaring success.
- [ ] Phase 4 end-to-end benchmark.
