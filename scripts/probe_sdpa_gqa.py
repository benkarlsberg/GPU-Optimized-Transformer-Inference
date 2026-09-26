#!/usr/bin/env python3
"""Why do the GQA SmolLM models launch more kernels per decode step?

Part A (operator level): for single-token decode shapes of each model, check
which SDPA backends accept the call (with ``enable_gqa`` for grouped-query
attention, as transformers passes it when no attention mask is needed) and
measure per-call kernel count, host time and wall time for the default
dispatch, and for the alternative of materialising K/V with ``repeat_kv``
before an ordinary multi-head SDPA call.

Part B (model level): rerun the decode-step measurement of
``profile_decode.py`` for the GQA models with transformers' GQA fast path
disabled (so K/V are repeated and SDPA sees multi-head shapes) and compare
with the default path.

Writes ``<out-dir>/sdpa_gqa_probe.json``.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_decode as pd  # noqa: E402
import transformers.integrations.sdpa_attention as hf_sdpa  # noqa: E402

from goti.backends.attention import configure_attention  # noqa: E402
from goti.benchmarks.workloads import SyntheticPrompt, make_input_ids  # noqa: E402
from goti.models.loader import load_causal_lm  # noqa: E402

SHAPES = {  # name: (query heads, kv heads, head_dim)
    "SmolLM-135M": (9, 3, 64),
    "SmolLM-360M": (15, 5, 64),
    "SmolLM-1.7B": (32, 32, 64),
}
BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}


def kernels_per_call(fn, n: int = 5) -> float:
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    k = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return len(k) / n


def time_call(fn, iters: int = 200) -> dict[str, float]:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    host = (time.perf_counter() - t0) / iters
    torch.cuda.synchronize()
    wall_unsynced = (time.perf_counter() - t0) / iters
    walls = []
    for _ in range(50):
        t1 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t1)
    return {"host_us": host * 1e6, "pipelined_us": wall_unsynced * 1e6,
            "synced_us": statistics.median(walls) * 1e6}


def repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


@torch.inference_mode()
def part_a(kv_len: int, batches: list[int]) -> list[dict]:
    out = []
    for name, (hq, hkv, d) in SHAPES.items():
        for b in batches:
            q = torch.randn(b, hq, 1, d, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(b, hkv, kv_len, d, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            gqa = hq != hkv
            kw = {"enable_gqa": True} if gqa else {}
            rec = {"model": name, "batch": b, "kv_len": kv_len, "gqa": gqa, "backends": {}}
            for bname, backend in BACKENDS.items():
                with warnings.catch_warnings(record=True) as caught, sdpa_kernel(backend):
                    warnings.simplefilter("always")
                    try:
                        F.scaled_dot_product_attention(q, k, v, **kw)
                        torch.cuda.synchronize()
                        rec["backends"][bname] = {"ok": True}
                    except RuntimeError:
                        reasons = [str(w.message).splitlines()[0][:200] for w in caught]
                        rec["backends"][bname] = {"ok": False, "reasons": reasons}
            default = lambda: F.scaled_dot_product_attention(q, k, v, **kw)  # noqa: E731
            rec["default"] = {"kernels_per_call": kernels_per_call(default), **time_call(default)}
            if gqa:
                n = hq // hkv
                rep = lambda: F.scaled_dot_product_attention(q, repeat_kv(k, n), repeat_kv(v, n))  # noqa: E731
                rec["repeat_kv_then_sdpa"] = {"kernels_per_call": kernels_per_call(rep), **time_call(rep)}
            out.append(rec)
            print(json.dumps({k_: rec[k_] for k_ in ("model", "batch")} | {
                "default": rec["default"], "repeat": rec.get("repeat_kv_then_sdpa"),
                "ok": {k_: v_["ok"] for k_, v_ in rec["backends"].items()}}), flush=True)
    return out


@torch.inference_mode()
def decode_cell(model_name: str, batch: int, gqa_fast_path: bool, args) -> dict:
    original = hf_sdpa.use_gqa_in_sdpa
    if not gqa_fast_path:
        hf_sdpa.use_gqa_in_sdpa = lambda *a, **k: False
    try:
        gc.collect()
        torch.cuda.empty_cache()
        loaded = load_causal_lm(model_name, dtype="bf16")
        configure_attention("sdpa")
        model = loaded.model
        ids = make_input_ids(loaded.tokenizer, SyntheticPrompt(batch_size=batch, prompt_length=args.prompt_length),
                             loaded.device)
        dec = pd.GreedyDecoder(model, ids, "dynamic", args.prompt_length + 128)
        synced = pd.run_synced(dec, 8, args.steps)
        unsynced = pd.run_unsynced(dec, 8, args.steps)
        prof = pd.run_profiled(dec, 8, 10, 2, int(model.config.num_hidden_layers), None)
        m = prof["median"] if prof.get("ok") else {}
        res = {
            "model": model_name, "batch": batch, "gqa_fast_path": gqa_fast_path,
            "wall_step_ms": pd.median(synced["step_ms"]),
            "dispatch_step_ms": unsynced["dispatch_per_step_ms"],
            "kernels_per_step": m.get("n_kernels"),
            "kernels_per_layer": m.get("kernels_per_layer"),
            "gpu_busy_step_ms": m.get("gpu_busy_us", 0) / 1e3,
            "gpu_us_per_layer": m.get("gpu_us_per_layer"),
            "top_ops_per_layer": m.get("layer_top_ops_mean"),
            "sdpa_kernels_layer0": sum(1 for x in (prof.get("layer0_kernel_sequence") or [])
                                       if x["op"] == "aten::scaled_dot_product_attention"),
        }
        del dec, model, loaded
        return res
    finally:
        hf_sdpa.use_gqa_in_sdpa = original
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kv-len", type=int, default=300)
    p.add_argument("--batches", default="1,32")
    p.add_argument("--prompt-length", type=int, default=256)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--model-cells", default="HuggingFaceTB/SmolLM-135M:1,32;HuggingFaceTB/SmolLM-360M:1,16")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--out-dir", type=Path, default=Path("results/profile_decode"))
    args = p.parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    result = {"environment": pd.environment(), "config": {k: str(v) for k, v in vars(args).items()}}
    result["operator_level"] = part_a(args.kv_len, batches)
    cells = []
    for part in args.model_cells.split(";"):
        name, bs = part.rsplit(":", 1)
        cells += [(name, int(b)) for b in bs.split(",")]
    model_level = []
    for pass_idx in range(args.passes):
        for name, b in cells:
            for fast in (True, False):
                r = decode_cell(name, b, fast, args)
                r["pass"] = pass_idx + 1
                model_level.append(r)
                print(json.dumps(r), flush=True)
    result["model_level"] = model_level
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sdpa_gqa_probe.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", args.out_dir / "sdpa_gqa_probe.json")


if __name__ == "__main__":
    main()
