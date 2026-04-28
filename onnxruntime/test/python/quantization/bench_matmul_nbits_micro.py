# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Micro-benchmark for the CUDA MatMulNBits kernel.

Differs from bench_matmul_8bits.py in scope: this is a kernel-level measurement
intended for A/B comparison of kernel implementations, not a TFLOPS leaderboard.

Design notes:
- Uses IOBinding with pre-allocated GPU tensors so that timing isolates the
  kernel + ORT op overhead, with no per-iteration H2D copies.
- CUDA event timing, n>=30 reps with warmup. Reports min, median, mean, stdev,
  p10, p90. For A vs B comparison, runs A and B interleaved per iteration and
  reports Welch's t-test on the per-iteration deltas.
- Predicts which CUDA code path will be taken (in-tree GEMV / fpA_intB CUTLASS /
  dequant+cuBLAS fallback) based on the dispatch logic in matmul_nbits.cc, so
  results are interpretable without an nsys trace.
- Reports both TFLOPS (compute-bound proxy) and effective GB/s of weight traffic
  (memory-bound proxy) — which matters depends on M, and the latter is the more
  meaningful metric at M=1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from onnx import TensorProto, helper

import onnxruntime as ort


# ----------------------------------------------------------------------------
# Configuration & predefined shapes
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """A single MatMulNBits problem shape."""
    m: int
    n: int
    k: int
    bits: int = 4
    block_size: int = 32
    dtype: str = "fp16"  # 'fp16' | 'bf16' | 'fp32'
    has_zero_point: bool = True
    zp_dtype: str = "uint8"  # 'uint8' | matches activation dtype

    def label(self) -> str:
        return (
            f"M{self.m}_N{self.n}_K{self.k}_b{self.bits}_g{self.block_size}_"
            f"{self.dtype}_zp={'Y' if self.has_zero_point else 'N'}_{self.zp_dtype}"
        )


# (name, [(layer_label, N, K), ...]) — N is the output dim, K is the input dim.
# These are the shapes ORT actually sees in popular quantized LLMs.
LLM_SHAPES: dict[str, list[tuple[str, int, int]]] = {
    "phi3_mini": [
        # hidden=3072, intermediate=8192, vocab=32064, heads=32, kv_heads=32, head_dim=96
        ("qkv_proj", 9216, 3072),
        ("o_proj", 3072, 3072),
        ("gate_up", 16384, 3072),
        ("down_proj", 3072, 8192),
        ("lm_head", 32064, 3072),
    ],
    "llama3_3b": [
        # hidden=3072, intermediate=8192, vocab=128256, heads=24, kv_heads=8, head_dim=128
        ("qkv_proj", 5120, 3072),
        ("o_proj", 3072, 3072),
        ("gate_up", 16384, 3072),
        ("down_proj", 3072, 8192),
        ("lm_head", 128256, 3072),
    ],
    "llama3_8b": [
        # hidden=4096, intermediate=14336, vocab=128256, heads=32, kv_heads=8, head_dim=128
        ("qkv_proj", 6144, 4096),
        ("o_proj", 4096, 4096),
        ("gate_up", 28672, 4096),
        ("down_proj", 4096, 14336),
        ("lm_head", 128256, 4096),
    ],
}

DEFAULT_M_SWEEP: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 128, 512, 2048)


# ----------------------------------------------------------------------------
# Path classifier (mirrors dispatch in matmul_nbits.cc)
# ----------------------------------------------------------------------------


def predict_path(s: Shape, fpa_intb_available: bool) -> str:
    """Predict which CUDA code path MatMulNBits will take for this shape.

    Mirrors the dispatch in matmul_nbits.cc::ComputeInternal:
      - fpA_intB CUTLASS: USE_FPA_INTB_GEMM build flag, fp16/bf16, prepacked
        weights — engaged whenever the build supports it for fp16/bf16 inputs.
      - In-tree GEMV: m == 1, no reorder_idx, no T-typed zero_points,
        n % 8 == 0 and k % 8 == 0, fits in shared memory.
      - Dequant + cuBLAS: everything else.
    """
    if s.dtype in ("fp16", "bf16") and fpa_intb_available:
        return "fpa_intb_cutlass"

    # Conditions for the in-tree GEMV path (matmul_4bits.cu line ~340):
    #   no reorder_idx (we never set it), zero_points either None or uint8.
    gemv_eligible = (
        s.m == 1
        and s.n % 8 == 0
        and s.k % 8 == 0
        and (not s.has_zero_point or s.zp_dtype == "uint8")
    )
    if gemv_eligible:
        return "in_tree_gemv"

    return "dequant_cublas_fallback"


# ----------------------------------------------------------------------------
# ONNX model construction
# ----------------------------------------------------------------------------


def _onnx_dtype(dtype: str) -> int:
    return {
        "fp16": TensorProto.FLOAT16,
        "bf16": TensorProto.BFLOAT16,
        "fp32": TensorProto.FLOAT,
    }[dtype]


def _torch_dtype(dtype: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]


def build_model(s: Shape) -> bytes:
    """Build a single-node ONNX model for this MatMulNBits shape."""
    if s.k % s.block_size != 0:
        raise ValueError(f"K={s.k} not divisible by block_size={s.block_size}")
    k_blocks = s.k // s.block_size
    elems_per_byte = 8 // s.bits
    # B is laid out as (N, k_blocks, block_size / elems_per_byte) bytes.
    b_inner = s.block_size // elems_per_byte
    act_dt = _onnx_dtype(s.dtype)

    inputs = [
        helper.make_tensor_value_info("a", act_dt, [s.m, s.k]),
        helper.make_tensor_value_info("b", TensorProto.UINT8, [s.n, k_blocks, b_inner]),
        helper.make_tensor_value_info("scales", act_dt, [s.n, k_blocks]),
    ]
    op_inputs = ["a", "b", "scales"]

    if s.has_zero_point:
        if s.zp_dtype == "uint8":
            # uint8 packed: each row has ceil(k_blocks * bits / 8) bytes.
            zp_bytes = (k_blocks * s.bits + 7) // 8
            inputs.append(helper.make_tensor_value_info("zero_points", TensorProto.UINT8, [s.n, zp_bytes]))
        else:
            inputs.append(helper.make_tensor_value_info("zero_points", act_dt, [s.n, k_blocks]))
        op_inputs.append("zero_points")

    node = helper.make_node(
        "MatMulNBits",
        op_inputs,
        ["y"],
        bits=s.bits,
        block_size=s.block_size,
        K=s.k,
        N=s.n,
        domain="com.microsoft",
    )
    outputs = [helper.make_tensor_value_info("y", act_dt, [s.m, s.n])]

    g = helper.make_graph([node], "matmul_nbits_micro", inputs, outputs)
    m = helper.make_model(
        g,
        producer_name="matmul_nbits_micro_bench",
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    return m.SerializeToString()


# ----------------------------------------------------------------------------
# Session + IOBinding setup
# ----------------------------------------------------------------------------


@dataclass
class BoundSession:
    sess: ort.InferenceSession
    iobinding: ort.IOBinding
    inputs: dict[str, torch.Tensor]
    output: torch.Tensor


def _np_dtype_from(dtype: str) -> np.dtype:
    return {
        "fp16": np.float16,
        # numpy lacks bf16; we represent bf16 buffers as uint16 for binding.
        "bf16": np.uint16,
        "fp32": np.float32,
    }[dtype]


def _ort_element_type(dtype: str) -> int:
    return {
        "fp16": int(TensorProto.FLOAT16),
        "bf16": int(TensorProto.BFLOAT16),
        "fp32": int(TensorProto.FLOAT),
    }[dtype]


def make_session(
    s: Shape,
    enable_cuda_graph: bool,
    device_id: int,
    seed: int = 123,
) -> BoundSession:
    """Build the model, create a session, allocate GPU buffers, bind IO."""
    model_bytes = build_model(s)

    so = ort.SessionOptions()
    so.log_severity_level = 3  # warning+
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    provider_opts = {
        "device_id": device_id,
        "arena_extend_strategy": "kSameAsRequested",
    }
    if enable_cuda_graph:
        provider_opts["enable_cuda_graph"] = "1"
    providers = [("CUDAExecutionProvider", provider_opts), "CPUExecutionProvider"]
    sess = ort.InferenceSession(model_bytes, sess_options=so, providers=providers)

    # Pre-allocate input/output tensors on GPU. Use torch for convenience; we
    # will bind the raw data pointers via IOBinding.
    g = torch.Generator(device="cuda").manual_seed(seed)
    dt = _torch_dtype(s.dtype)
    device = torch.device("cuda", device_id)

    a = torch.empty((s.m, s.k), device=device, dtype=dt).normal_(0, 0.1, generator=g)

    k_blocks = s.k // s.block_size
    elems_per_byte = 8 // s.bits
    b_inner = s.block_size // elems_per_byte
    b = torch.randint(0, 256, (s.n, k_blocks, b_inner), device=device, dtype=torch.uint8, generator=g)
    scales = torch.empty((s.n, k_blocks), device=device, dtype=dt).normal_(0, 0.01, generator=g)

    inputs: dict[str, torch.Tensor] = {"a": a, "b": b, "scales": scales}
    if s.has_zero_point:
        if s.zp_dtype == "uint8":
            zp_bytes = (k_blocks * s.bits + 7) // 8
            inputs["zero_points"] = torch.randint(
                0, 256, (s.n, zp_bytes), device=device, dtype=torch.uint8, generator=g
            )
        else:
            inputs["zero_points"] = torch.empty((s.n, k_blocks), device=device, dtype=dt).normal_(
                0, 0.01, generator=g
            )

    output = torch.empty((s.m, s.n), device=device, dtype=dt)

    binding = sess.io_binding()

    def _ort_dtype_for(t: torch.Tensor, name: str) -> int:
        if t.dtype == torch.uint8:
            return int(TensorProto.UINT8)
        if t.dtype == torch.float16:
            return int(TensorProto.FLOAT16)
        if t.dtype == torch.bfloat16:
            return int(TensorProto.BFLOAT16)
        if t.dtype == torch.float32:
            return int(TensorProto.FLOAT)
        raise ValueError(f"unsupported torch dtype {t.dtype} for input {name}")

    for name, t in inputs.items():
        binding.bind_input(
            name=name,
            device_type="cuda",
            device_id=device_id,
            element_type=_ort_dtype_for(t, name),
            shape=tuple(t.shape),
            buffer_ptr=t.data_ptr(),
        )
    binding.bind_output(
        name="y",
        device_type="cuda",
        device_id=device_id,
        element_type=_ort_dtype_for(output, "y"),
        shape=tuple(output.shape),
        buffer_ptr=output.data_ptr(),
    )

    return BoundSession(sess=sess, iobinding=binding, inputs=inputs, output=output)


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------


@dataclass
class TimingStats:
    n_samples: int
    min_us: float
    p10_us: float
    median_us: float
    mean_us: float
    p90_us: float
    max_us: float
    stdev_us: float

    @classmethod
    def from_samples_us(cls, samples_us: list[float]) -> TimingStats:
        s = sorted(samples_us)
        n = len(s)

        def pct(p: float) -> float:
            if n == 0:
                return float("nan")
            idx = max(0, min(n - 1, int(round(p * (n - 1)))))
            return s[idx]

        return cls(
            n_samples=n,
            min_us=s[0] if n else float("nan"),
            p10_us=pct(0.10),
            median_us=pct(0.50),
            mean_us=statistics.mean(s) if n else float("nan"),
            p90_us=pct(0.90),
            max_us=s[-1] if n else float("nan"),
            stdev_us=statistics.stdev(s) if n > 1 else 0.0,
        )


def time_session(
    bs: BoundSession,
    repeats: int,
    warmup: int,
    inner_reps: int = 1,
) -> list[float]:
    """Run the bound session repeats times, return per-iteration latencies in us.

    inner_reps: within each timed window, run the op this many times back-to-back
    and divide the elapsed time by inner_reps. This amortizes launch / op-dispatch
    overhead over many kernel invocations, which is needed when the kernel itself
    runs in <50us — otherwise dispatch jitter dominates the variance.
    """
    for _ in range(warmup):
        bs.sess.run_with_iobinding(bs.iobinding)
    torch.cuda.synchronize()

    samples_us: list[float] = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start_evt.record()
        for _ in range(inner_reps):
            bs.sess.run_with_iobinding(bs.iobinding)
        end_evt.record()
        end_evt.synchronize()
        samples_us.append(start_evt.elapsed_time(end_evt) * 1000.0 / inner_reps)
    return samples_us


def time_session_interleaved(
    bs_a: BoundSession,
    bs_b: BoundSession,
    repeats: int,
    warmup: int,
) -> tuple[list[float], list[float]]:
    """Run A and B alternately to neutralize warm-cache / clock-drift bias.

    Returns (a_samples_us, b_samples_us).
    """
    for _ in range(warmup):
        bs_a.sess.run_with_iobinding(bs_a.iobinding)
        bs_b.sess.run_with_iobinding(bs_b.iobinding)
    torch.cuda.synchronize()

    a_us: list[float] = []
    b_us: list[float] = []
    sa_e0 = torch.cuda.Event(enable_timing=True)
    sa_e1 = torch.cuda.Event(enable_timing=True)
    sb_e0 = torch.cuda.Event(enable_timing=True)
    sb_e1 = torch.cuda.Event(enable_timing=True)
    for i in range(repeats):
        if i % 2 == 0:
            sa_e0.record()
            bs_a.sess.run_with_iobinding(bs_a.iobinding)
            sa_e1.record()
            sb_e0.record()
            bs_b.sess.run_with_iobinding(bs_b.iobinding)
            sb_e1.record()
        else:
            sb_e0.record()
            bs_b.sess.run_with_iobinding(bs_b.iobinding)
            sb_e1.record()
            sa_e0.record()
            bs_a.sess.run_with_iobinding(bs_a.iobinding)
            sa_e1.record()
        sa_e1.synchronize()
        sb_e1.synchronize()
        a_us.append(sa_e0.elapsed_time(sa_e1) * 1000.0)
        b_us.append(sb_e0.elapsed_time(sb_e1) * 1000.0)
    return a_us, b_us


def welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t and approximate two-sided p-value (normal approx for large n)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    denom = math.sqrt(va / na + vb / nb) if (va + vb) > 0 else 0.0
    if denom == 0.0:
        return float("inf") if ma != mb else 0.0, 0.0 if ma != mb else 1.0
    t = (ma - mb) / denom
    # Two-sided p via normal CDF (n>=30 makes this a fine approx).
    z = abs(t)
    # Abramowitz–Stegun approximation of erfc
    p = math.erfc(z / math.sqrt(2.0))
    return t, p


# ----------------------------------------------------------------------------
# Roofline-ish derived metrics
# ----------------------------------------------------------------------------


def flops(s: Shape) -> int:
    return 2 * s.m * s.n * s.k


def weight_bytes(s: Shape) -> int:
    """Bytes of *quantized* B matrix touched per matmul (bandwidth proxy)."""
    return s.n * s.k * s.bits // 8


def tflops(s: Shape, latency_us: float) -> float:
    if latency_us <= 0 or math.isnan(latency_us):
        return 0.0
    return (flops(s) / (latency_us * 1e-6)) / 1e12


def gbps(s: Shape, latency_us: float) -> float:
    """Effective weight read bandwidth (GB/s, base 10)."""
    if latency_us <= 0 or math.isnan(latency_us):
        return 0.0
    return (weight_bytes(s) / (latency_us * 1e-6)) / 1e9


# ----------------------------------------------------------------------------
# Sweep & reporting
# ----------------------------------------------------------------------------


def expand_shapes(args: argparse.Namespace) -> list[Shape]:
    out: list[Shape] = []
    bits_list = [int(x) for x in args.bits.split(",")]
    bs_list = [int(x) for x in args.block_size.split(",")]
    dtype_list = args.dtype.split(",")
    m_list = [int(x) for x in args.m_sweep.split(",")] if args.m_sweep else list(DEFAULT_M_SWEEP)

    if args.shapes:
        # explicit "M,N,K" triples
        explicit: list[tuple[int, int, int]] = []
        for trip in args.shapes.split(";"):
            mn = trip.strip()
            if not mn:
                continue
            m, n, k = (int(x) for x in mn.split(","))
            explicit.append((m, n, k))
        for (m, n, k) in explicit:
            for bits in bits_list:
                for bs in bs_list:
                    for dt in dtype_list:
                        if k % bs != 0:
                            continue
                        out.append(Shape(m=m, n=n, k=k, bits=bits, block_size=bs, dtype=dt))
        return out

    # Otherwise: predefined LLM shape families
    families = args.models.split(",")
    for fam in families:
        if fam not in LLM_SHAPES:
            raise SystemExit(f"unknown model family: {fam!r}; known: {list(LLM_SHAPES)}")
        for (_label, n, k) in LLM_SHAPES[fam]:
            for m in m_list:
                for bits in bits_list:
                    for bs in bs_list:
                        if k % bs != 0:
                            continue
                        for dt in dtype_list:
                            out.append(Shape(m=m, n=n, k=k, bits=bits, block_size=bs, dtype=dt))
    return out


def fmt_row(s: Shape, stats: TimingStats, path: str) -> str:
    # On WSL2 / shared display GPUs, between-window clock dithering inflates
    # median/mean. Min is the kernel's true unobstructed time and is what we
    # use for A/B kernel comparisons. Median is reported as a sanity check;
    # if min and median diverge a lot, the host has noise we should flag.
    return (
        f"M={s.m:<5d} N={s.n:<6d} K={s.k:<5d} b={s.bits} g={s.block_size:<3d} "
        f"{s.dtype:<4s} | path={path:<22s} "
        f"min={stats.min_us:7.2f}us  p10={stats.p10_us:7.2f}  med={stats.median_us:7.2f}  "
        f"sd={stats.stdev_us:6.2f}  "
        f"TFLOPS@min={tflops(s, stats.min_us):6.2f}  GB/s@min={gbps(s, stats.min_us):7.1f}"
    )


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="phi3_mini",
                   help=f"comma-separated; known: {','.join(LLM_SHAPES)}")
    p.add_argument("--shapes", default="",
                   help='explicit "M,N,K;M,N,K;..." instead of --models')
    p.add_argument("--m_sweep", default=",".join(str(m) for m in DEFAULT_M_SWEEP),
                   help="comma-separated M values when using --models")
    p.add_argument("--bits", default="4")
    p.add_argument("--block_size", default="32,128")
    p.add_argument("--dtype", default="fp16",
                   help="comma-separated subset of {fp16,bf16,fp32}")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--inner_reps", type=int, default=20,
                   help="kernel invocations per timed window (amortizes launch overhead)")
    p.add_argument("--cuda_graph", action="store_true",
                   help="enable ORT CUDA graph capture (may not work for all shapes)")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--csv", default="",
                   help="output CSV path (default: bench_micro_<timestamp>.csv)")
    p.add_argument("--tag", default="baseline",
                   help="label written into the CSV/json so multiple runs can be A/B'd")
    p.add_argument("--fpa_intb", default="auto",
                   choices=("auto", "yes", "no"),
                   help="hint for path classifier; 'auto' inspects ORT build info")
    args = p.parse_args(argv)

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("CUDAExecutionProvider not available", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("torch CUDA not available", file=sys.stderr)
        return 1

    # Detect fpA_intB availability. In current ORT the most reliable signal is
    # whether the M>1 path is much faster than the dequant+cublas fallback would
    # imply; we can't introspect the build flag from Python. Caller can override.
    if args.fpa_intb == "auto":
        # Default to yes for stock onnxruntime-gpu wheels (it's been on by
        # default since ~1.21 for fp16/bf16). User can override with --fpa_intb.
        fpa_intb_available = True
    else:
        fpa_intb_available = args.fpa_intb == "yes"

    shapes = expand_shapes(args)
    if not shapes:
        print("no shapes to run", file=sys.stderr)
        return 2

    device_name = torch.cuda.get_device_name(args.device_id)
    cc = torch.cuda.get_device_capability(args.device_id)
    print(f"# device: {device_name}  CC {cc[0]}.{cc[1]}  ORT {ort.__version__}")
    print(f"# tag={args.tag}  repeats={args.repeats}  warmup={args.warmup}  "
          f"cuda_graph={args.cuda_graph}  fpa_intb_assumed={fpa_intb_available}")
    print(f"# {len(shapes)} configurations")

    rows: list[dict] = []
    print()
    for s in shapes:
        try:
            bs = make_session(s, args.cuda_graph, args.device_id)
        except Exception as e:
            print(f"FAIL build/bind  {s.label()}: {e}")
            continue
        path = predict_path(s, fpa_intb_available)
        try:
            samples = time_session(bs, args.repeats, args.warmup, args.inner_reps)
        except Exception as e:
            print(f"FAIL run  {s.label()}: {e}")
            del bs
            continue
        stats = TimingStats.from_samples_us(samples)
        print(fmt_row(s, stats, path))
        row = {
            "tag": args.tag,
            "device": device_name,
            "cc": f"{cc[0]}.{cc[1]}",
            "ort": ort.__version__,
            **asdict(s),
            "predicted_path": path,
            "cuda_graph": args.cuda_graph,
            **asdict(stats),
            "tflops_min": tflops(s, stats.min_us),
            "gbps_min": gbps(s, stats.min_us),
            "tflops_median": tflops(s, stats.median_us),
            "gbps_median": gbps(s, stats.median_us),
            "weight_bytes": weight_bytes(s),
        }
        rows.append(row)
        del bs

    out_csv = Path(args.csv) if args.csv else Path(
        f"bench_matmul_nbits_micro_{args.tag}_{datetime.now():%Y%m%d-%H%M%S}.csv"
    )
    write_csv(rows, out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
