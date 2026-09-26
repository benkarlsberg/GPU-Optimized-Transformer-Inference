#!/usr/bin/env python3
"""Per-step profile of eager transformer decoding.

For every (model, cache, batch) cell this script prefills the synthetic prompt
used by the benchmark harness, then runs a manual greedy decode loop over a KV
cache and measures, per decode step:

  * wall time, synchronising the device after every step;
  * CPU dispatch time, i.e. how long the Python/framework side takes to return
    from a step when the device is *not* synchronised between steps;
  * torch.profiler (CPU + CUDA) statistics: kernel launches, GPU busy time
    (union of kernel/memcpy/memset intervals), GPU idle fraction, top kernels,
    host time in aten ops and in CUDA runtime calls, and a per-decoder-layer
    breakdown obtained from forward hooks;
  * achieved weight (and weight + KV-cache) bandwidth over GPU busy time.

It also times ``model.generate`` end to end for the same prompt so the manual
loop can be checked against the harness' per-token latency.

Outputs (under --out-dir): ``raw.json`` (per-step data for every pass),
``summary.json`` (per-cell aggregates across passes) and, optionally, gzipped
Chrome traces of the profiled steps from the first pass.
"""
from __future__ import annotations

import argparse
import bisect
import gc
import gzip
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import DynamicCache, StaticCache

from goti.backends.attention import configure_attention
from goti.benchmarks.workloads import SyntheticPrompt, make_input_ids
from goti.models.loader import load_causal_lm

STEP_TAG = "decode_step"
LAYER_TAG = "decoder_layer"
SYNC_APIS = {
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaMemcpy",
    "cudaMemcpyAsync",
}
DEFAULT_CELLS = (
    "HuggingFaceTB/SmolLM-135M:dynamic:1,32,128;"
    "HuggingFaceTB/SmolLM-360M:dynamic:1,16,64;"
    "HuggingFaceTB/SmolLM-1.7B:dynamic:1,16,32;"
    "HuggingFaceTB/SmolLM-1.7B:static:1,32"
)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def sync() -> None:
    torch.cuda.synchronize()


def median(xs: list[float]) -> float | None:
    return float(statistics.median(xs)) if xs else None


def union_length(intervals: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def short_kernel(name: str, n: int = 110) -> str:
    name = name.replace("void ", "", 1)
    return name if len(name) <= n else name[:n] + "..."


def sanitize_path(path: str) -> str:
    """Drop machine-specific prefixes from a source path."""
    p = path.replace("\\", "/")
    for marker in ("site-packages/", "/src/", "/scripts/"):
        if marker in p:
            return p.split(marker, 1)[1]
    return os.path.basename(p)


def os_name() -> str:
    if platform.system() == "Windows":
        try:
            build = int(platform.version().split(".")[-1])
        except ValueError:
            build = 0
        return f"Windows {'11' if build >= 22000 else platform.release()} (build {build})"
    return f"{platform.system()} {platform.release()}"


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "os": os_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "sm": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10,
        )
        env["driver"] = out.strip().splitlines()[0]
    except Exception:
        env["driver"] = None
    return env


def decoder_layers(model) -> list[torch.nn.Module]:
    for path in ("model.layers", "transformer.h"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers")


# ----------------------------------------------------------------------------
# manual decode loop
# ----------------------------------------------------------------------------
class GreedyDecoder:
    """Prefill + one-token-at-a-time greedy decoding over a KV cache.

    Mirrors what ``generate`` does per step for greedy search: feed the last
    token with the grown attention mask and cache position, copy the last-row
    logits to float32 and take the argmax.
    """

    def __init__(self, model, input_ids: torch.Tensor, cache_impl: str, max_cache_len: int):
        self.model = model
        self.input_ids = input_ids
        self.cache_impl = cache_impl
        self.max_cache_len = max_cache_len

    def _new_cache(self):
        if self.cache_impl == "static":
            return StaticCache(config=self.model.config, max_cache_len=self.max_cache_len)
        return DynamicCache(config=self.model.config)

    def prefill(self) -> None:
        b, length = self.input_ids.shape
        self.cache = self._new_cache()
        self.mask = torch.ones_like(self.input_ids)
        self.cache_position = torch.arange(length, device=self.input_ids.device)
        out = self.model(
            input_ids=self.input_ids,
            attention_mask=self.mask,
            past_key_values=self.cache,
            cache_position=self.cache_position,
            use_cache=True,
            logits_to_keep=1,
        )
        self.tok = out.logits[:, -1, :].to(copy=True, dtype=torch.float32).argmax(-1, keepdim=True)

    def step(self) -> None:
        b = self.tok.shape[0]
        self.mask = torch.cat([self.mask, self.mask.new_ones((b, 1))], dim=-1)
        self.cache_position = self.cache_position[-1:] + 1
        out = self.model(
            input_ids=self.tok,
            attention_mask=self.mask,
            past_key_values=self.cache,
            cache_position=self.cache_position,
            use_cache=True,
        )
        logits = out.logits[:, -1, :].to(copy=True, dtype=torch.float32)
        self.tok = logits.argmax(-1, keepdim=True)

    def kv_len(self) -> int:
        return int(self.mask.shape[-1])


def run_synced(dec: GreedyDecoder, warmup: int, steps: int) -> dict[str, Any]:
    sync()
    t0 = time.perf_counter()
    dec.prefill()
    sync()
    prefill_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        dec.step()
    sync()
    kv_start = dec.kv_len()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        dec.step()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return {"prefill_ms": prefill_ms, "step_ms": times, "kv_len_start": kv_start, "kv_len_end": dec.kv_len()}


def run_unsynced(dec: GreedyDecoder, warmup: int, steps: int) -> dict[str, Any]:
    # the first forward after loading pays one-off library initialisation, so the
    # prefill is timed here (second run) rather than in run_synced
    sync()
    t0 = time.perf_counter()
    dec.prefill()
    sync()
    prefill_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        dec.step()
    sync()
    marks = []
    t0 = time.perf_counter()
    for _ in range(steps):
        dec.step()
        marks.append(time.perf_counter())
    t_cpu = marks[-1] - t0
    sync()
    t_wall = time.perf_counter() - t0
    per_step = [(b - a) * 1e3 for a, b in zip([t0] + marks[:-1], marks)]
    return {
        "prefill_ms": prefill_ms,
        "dispatch_step_ms": per_step,
        "dispatch_total_ms": t_cpu * 1e3,
        "wall_total_ms": t_wall * 1e3,
        "dispatch_per_step_ms": t_cpu * 1e3 / steps,
        "wall_per_step_ms": t_wall * 1e3 / steps,
    }


def count_host_syncs(dec: GreedyDecoder, warmup: int) -> dict[str, Any]:
    """Count synchronising CUDA calls issued during one decode step."""
    dec.prefill()
    for _ in range(warmup):
        dec.step()
    sync()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            dec.step()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    sync()
    # the call that switches the debug mode back is itself reported; ignore it
    hits = [w for w in caught if "synchroniz" in str(w.message).lower()
            and not sanitize_path(w.filename).startswith("torch/cuda/")]
    sites: dict[str, int] = {}
    for w in hits:
        key = f"{sanitize_path(w.filename)}:{w.lineno}"
        sites[key] = sites.get(key, 0) + 1
    return {"syncs_per_step": len(hits), "sites": sites}


def run_profiled(dec: GreedyDecoder, warmup: int, steps: int, skip: int, n_layers: int,
                 trace_out: Path | None) -> dict[str, Any]:
    dec.prefill()
    for _ in range(warmup):
        dec.step()
    sync()

    stack: list[Any] = []

    def pre_hook(_mod, _args):
        rf = torch.profiler.record_function(LAYER_TAG)
        rf.__enter__()
        stack.append(rf)

    def post_hook(_mod, _args, _out):
        stack.pop().__exit__(None, None, None)

    handles = []
    for layer in decoder_layers(dec.model):
        handles.append(layer.register_forward_pre_hook(pre_hook))
        handles.append(layer.register_forward_hook(post_hook))
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    walls = []
    try:
        with torch.profiler.profile(activities=activities, record_shapes=False, with_stack=False) as prof:
            for _ in range(skip + steps):
                t0 = time.perf_counter()
                with torch.profiler.record_function(STEP_TAG):
                    dec.step()
                    sync()
                walls.append((time.perf_counter() - t0) * 1e3)
    finally:
        for h in handles:
            h.remove()

    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(tmp)
        with open(tmp, encoding="utf-8") as f:
            trace = json.load(f)
        if trace_out is not None:
            trace_out.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "rb") as src, gzip.open(trace_out, "wb", compresslevel=6) as dst:
                dst.write(src.read())
    finally:
        os.remove(tmp)
    analysis = analyze_trace(trace, n_layers=n_layers, skip=skip)
    analysis["profiled_wall_step_ms_perf_counter"] = walls[skip:]
    return analysis


def analyze_trace(trace: dict[str, Any], n_layers: int, skip: int) -> dict[str, Any]:
    evs = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X" and "dur" in e]
    step_evs = sorted(
        (e for e in evs if e.get("cat") == "user_annotation" and e.get("name") == STEP_TAG),
        key=lambda e: e["ts"],
    )
    if not step_evs:
        return {"ok": False, "reason": "no step annotations in trace"}
    main_pid, main_tid = step_evs[0]["pid"], step_evs[0]["tid"]
    step_evs = step_evs[skip:]
    layer_rng = sorted(
        (e["ts"], e["ts"] + e["dur"]) for e in evs
        if e.get("cat") == "user_annotation" and e.get("name") == LAYER_TAG
    )
    kernels = [e for e in evs if e.get("cat") == "kernel"]
    memops = [e for e in evs if e.get("cat") in ("gpu_memcpy", "gpu_memset")]
    api = sorted((e for e in evs if e.get("cat") in ("cuda_runtime", "cuda_driver")), key=lambda e: e["ts"])
    if not kernels:
        return {"ok": False, "reason": "no CUDA kernel events in trace (CUPTI activity unavailable?)"}
    launch_ts: dict[Any, float] = {}
    for e in api:
        c = (e.get("args") or {}).get("correlation")
        if c is not None:
            launch_ts[c] = e["ts"]

    def lts(e):
        return launch_ts.get((e.get("args") or {}).get("correlation"), e["ts"])

    cpu_ops = sorted(
        (e["ts"], e["ts"] + e["dur"], e["name"]) for e in evs
        if e.get("cat") == "cpu_op" and e.get("pid") == main_pid and e.get("tid") == main_tid
    )
    top_ops: list[tuple[float, float, str]] = []
    end = float("-inf")
    for s, e_, n in cpu_ops:
        if s >= end:
            top_ops.append((s, e_, n))
            end = e_
    top_starts = [t[0] for t in top_ops]

    def top_op_at(ts: float) -> str | None:
        i = bisect.bisect_right(top_starts, ts) - 1
        if i >= 0 and top_ops[i][0] <= ts <= top_ops[i][1]:
            return top_ops[i][2]
        return None

    # (launch_ts, start, end, name, is_kernel)
    gpu = sorted(
        [(lts(k), k["ts"], k["ts"] + k["dur"], k["name"], True) for k in kernels]
        + [(lts(m), m["ts"], m["ts"] + m["dur"], m["name"], False) for m in memops]
    )
    gpu_launch = [g[0] for g in gpu]

    def gpu_in(s: float, e: float):
        return gpu[bisect.bisect_left(gpu_launch, s): bisect.bisect_right(gpu_launch, e)]

    per_step = []
    kernel_totals: dict[str, list[float]] = {}
    layer0: list[dict[str, Any]] | None = None
    layer0_ops: list[str] | None = None
    for st in step_evs:
        s, e = st["ts"], st["ts"] + st["dur"]
        g = gpu_in(s, e)
        ks = [x for x in g if x[4]]
        busy = union_length([(x[1], x[2]) for x in g])
        kernel_sum = sum(x[2] - x[1] for x in ks)
        span = (max(x[2] for x in g) - min(x[1] for x in g)) if g else 0.0
        lr = [r for r in layer_rng if s <= r[0] <= e]
        layer_k, layer_gpu, layer_busy, layer_cpu, layer_ops = [], [], [], [], []
        for li, (ls, le) in enumerate(lr):
            lg = [x for x in gpu_in(ls, le) if x[4]]
            layer_k.append(len(lg))
            layer_gpu.append(sum(x[2] - x[1] for x in lg))
            layer_busy.append(union_length([(x[1], x[2]) for x in lg]))
            layer_cpu.append(le - ls)
            ops_here = [t for t in top_ops if ls <= t[0] <= le]
            layer_ops.append(len(ops_here))
            if layer0 is None and li == 0:
                layer0 = [{"op": top_op_at(x[0]), "kernel": short_kernel(x[3]),
                           "gpu_us": round(x[2] - x[1], 2)} for x in lg]
                layer0_ops = [t[2] for t in ops_here]
        a = [x for x in api if s <= x["ts"] <= e]
        launches = [x for x in a if "Launch" in x["name"]]
        syncs = [x for x in a if x["name"] in SYNC_APIS]
        tail = syncs[-1]["dur"] if syncs and syncs[-1]["name"] == "cudaDeviceSynchronize" else 0.0
        in_op_syncs = [x for x in syncs if top_op_at(x["ts"]) is not None]
        tops = [t for t in top_ops if s <= t[0] <= e]
        wall = e - s
        sync_total = sum(x["dur"] for x in syncs)
        for x in ks:
            kernel_totals.setdefault(x[3], []).append(x[2] - x[1])
        per_step.append({
            "wall_us": wall,
            "gpu_busy_us": busy,
            "kernel_sum_us": kernel_sum,
            "gpu_span_us": span,
            "n_kernels": len(ks),
            "n_memops": len(g) - len(ks),
            "n_launch_calls": len(launches),
            "launch_api_us": sum(x["dur"] for x in launches),
            "sync_calls": len(syncs),
            "sync_in_ops_calls": len(in_op_syncs),
            "sync_in_ops_us": sum(x["dur"] for x in in_op_syncs),
            "sync_wait_us": sync_total,
            "tail_sync_us": tail,
            "host_busy_us": wall - sync_total,
            "n_top_ops": len(tops),
            "aten_top_ops_us": union_length([(t[0], t[1]) for t in tops]),
            "n_layers_seen": len(lr),
            "layer_kernels_total": sum(layer_k),
            "layer_gpu_us_total": sum(layer_gpu),
            "layer_busy_us_total": sum(layer_busy),
            "layer_cpu_us_mean": statistics.mean(layer_cpu) if layer_cpu else None,
            "layer_top_ops_mean": statistics.mean(layer_ops) if layer_ops else None,
        })

    n_steps = len(per_step)

    def med(key):
        return median([p[key] for p in per_step])

    top = sorted(kernel_totals.items(), key=lambda kv: -sum(kv[1]))[:12]
    total_k = sum(sum(v) for v in kernel_totals.values())
    top_kernels = [{
        "kernel": short_kernel(k),
        "us_per_step": sum(v) / n_steps,
        "calls_per_step": len(v) / n_steps,
        "share": sum(v) / total_k if total_k else None,
    } for k, v in top]
    summary = {k: med(k) for k in per_step[0].keys()}
    summary["kernels_per_layer"] = summary["layer_kernels_total"] / n_layers
    summary["kernels_outside_layers"] = summary["n_kernels"] - summary["layer_kernels_total"]
    summary["gpu_us_per_layer"] = summary["layer_gpu_us_total"] / n_layers
    summary["gpu_idle_fraction"] = 1.0 - summary["gpu_busy_us"] / summary["wall_us"]
    return {
        "ok": True,
        "n_steps": n_steps,
        "median": summary,
        "per_step": per_step,
        "top_kernels": top_kernels,
        "layer0_kernel_sequence": layer0,
        "layer0_top_ops": layer0_ops,
    }


def run_generate(model, tokenizer, input_ids, cache_impl: str, max_new: int,
                 warmup: int, iters: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cache_impl == "static":
        kwargs = {"cache_implementation": "static", "disable_compile": True}
    attn = torch.ones_like(input_ids)
    e2e, gen = [], None
    for i in range(warmup + iters):
        sync()
        t0 = time.perf_counter()
        out = model.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=max_new,
                             do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, **kwargs)
        sync()
        dt = (time.perf_counter() - t0) * 1e3
        gen = int(out.shape[-1] - input_ids.shape[-1])
        if i >= warmup:
            e2e.append(dt)
        del out
    return {"e2e_ms": e2e, "generated_tokens": gen, "max_new_tokens": max_new}


# ----------------------------------------------------------------------------
# cell driver
# ----------------------------------------------------------------------------
@torch.inference_mode()
def profile_cell(model_name: str, cache_impl: str, batch: int, args, trace_out: Path | None) -> dict[str, Any]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_cell = time.perf_counter()
    loaded = load_causal_lm(model_name, dtype=args.dtype)
    configure_attention(args.attention)
    model, tok = loaded.model, loaded.tokenizer
    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    kv_heads = int(getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads)
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    elem = next(model.parameters()).element_size()
    info = {
        "model": model_name,
        "cache": cache_impl,
        "batch_size": batch,
        "attn_implementation": getattr(cfg, "_attn_implementation", None),
        "num_layers": n_layers,
        "hidden_size": int(cfg.hidden_size),
        "num_heads": int(cfg.num_attention_heads),
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate_size": int(getattr(cfg, "intermediate_size", 0)),
        "vocab_size": int(cfg.vocab_size),
        "tie_word_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
        "weight_bytes": weight_bytes,
    }
    input_ids = make_input_ids(tok, SyntheticPrompt(batch_size=batch, prompt_length=args.prompt_length), loaded.device)
    max_cache_len = args.prompt_length + args.max_new_tokens
    dec = GreedyDecoder(model, input_ids, cache_impl, max_cache_len)
    try:
        synced = run_synced(dec, args.warmup_steps, args.steps)
        unsynced = run_unsynced(dec, args.warmup_steps, args.steps)
        syncs = count_host_syncs(dec, args.warmup_steps)
        prof = run_profiled(dec, args.warmup_steps, args.profile_steps, args.profile_skip, n_layers, trace_out)
        del dec
        gen = run_generate(model, tok, input_ids, cache_impl, args.max_new_tokens,
                           args.generate_warmup, args.generate_iters)
    except torch.cuda.OutOfMemoryError as exc:
        del model, loaded
        gc.collect()
        torch.cuda.empty_cache()
        return {**info, "oom": True, "error": str(exc).splitlines()[0]}

    kv_mid = (synced["kv_len_start"] + synced["kv_len_end"]) / 2
    kv_read_len = max_cache_len if cache_impl == "static" else kv_mid
    kv_bytes = 2 * n_layers * kv_heads * head_dim * kv_read_len * batch * elem
    wall_ms = median(synced["step_ms"])
    derived: dict[str, Any] = {
        "wall_step_ms": wall_ms,
        "dispatch_step_ms": unsynced["dispatch_per_step_ms"],
        "unsynced_wall_step_ms": unsynced["wall_per_step_ms"],
        "dispatch_over_wall": unsynced["dispatch_per_step_ms"] / wall_ms,
        "prefill_ms": unsynced["prefill_ms"],
        "kv_len_mid": kv_mid,
        "kv_bytes_read_per_step": kv_bytes,
        "generate_e2e_ms": median(gen["e2e_ms"]),
        "generate_ms_per_token": median(gen["e2e_ms"]) / args.max_new_tokens,
        "generate_decode_ms_per_step_est": (median(gen["e2e_ms"]) - unsynced["prefill_ms"]) / max(1, (gen["generated_tokens"] or 1) - 1),
        "syncs_per_step": syncs["syncs_per_step"],
    }
    if prof.get("ok"):
        m = prof["median"]
        busy_s = m["gpu_busy_us"] * 1e-6
        derived.update({
            "profiled_wall_step_ms": m["wall_us"] / 1e3,
            "gpu_busy_step_ms": m["gpu_busy_us"] / 1e3,
            "gpu_idle_fraction_profiled": m["gpu_idle_fraction"],
            "gpu_idle_fraction_vs_unprofiled_wall": 1.0 - (m["gpu_busy_us"] / 1e3) / wall_ms,
            "kernels_per_step": m["n_kernels"],
            "kernels_per_layer": m["kernels_per_layer"],
            "kernels_outside_layers": m["kernels_outside_layers"],
            "gpu_us_per_layer": m["gpu_us_per_layer"],
            "cpu_us_per_layer_profiled": m["layer_cpu_us_mean"],
            "top_ops_per_layer": m["layer_top_ops_mean"],
            "host_busy_step_ms_profiled": m["host_busy_us"] / 1e3,
            "aten_top_ops_step_ms_profiled": m["aten_top_ops_us"] / 1e3,
            "launch_api_step_ms_profiled": m["launch_api_us"] / 1e3,
            "weight_bw_GBps": weight_bytes / busy_s / 1e9,
            "weight_plus_kv_bw_GBps": (weight_bytes + kv_bytes) / busy_s / 1e9,
            "weight_bw_frac_of_rated": weight_bytes / busy_s / args.rated_bw,
            "weight_plus_kv_bw_frac_of_rated": (weight_bytes + kv_bytes) / busy_s / args.rated_bw,
        })
    out = {
        **info,
        "oom": False,
        "max_cache_len_static": max_cache_len if cache_impl == "static" else None,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "derived": derived,
        "synced": synced,
        "unsynced": unsynced,
        "host_syncs": syncs,
        "profile": prof,
        "generate": gen,
        "cell_seconds": time.perf_counter() - t_cell,
    }
    del model, loaded
    gc.collect()
    torch.cuda.empty_cache()
    return out


def parse_cells(spec: str) -> list[tuple[str, str, int]]:
    cells = []
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        name, cache, batches = part.rsplit(":", 2)
        for b in batches.split(","):
            cells.append((name, cache, int(b)))
    return cells


def cell_key(c: dict[str, Any]) -> str:
    return f"{c['model'].split('/')[-1]}|{c['cache']}|b{c['batch_size']}"


SUMMARY_KEYS = [
    "wall_step_ms", "dispatch_step_ms", "unsynced_wall_step_ms", "dispatch_over_wall",
    "profiled_wall_step_ms", "gpu_busy_step_ms", "gpu_idle_fraction_profiled",
    "gpu_idle_fraction_vs_unprofiled_wall", "kernels_per_step", "kernels_per_layer",
    "kernels_outside_layers", "gpu_us_per_layer", "cpu_us_per_layer_profiled", "top_ops_per_layer",
    "host_busy_step_ms_profiled", "aten_top_ops_step_ms_profiled", "launch_api_step_ms_profiled",
    "weight_bw_GBps", "weight_plus_kv_bw_GBps", "weight_bw_frac_of_rated",
    "weight_plus_kv_bw_frac_of_rated", "prefill_ms", "generate_e2e_ms", "generate_ms_per_token",
    "generate_decode_ms_per_step_est", "syncs_per_step",
]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by.setdefault(cell_key(r), []).append(r)
    cells = {}
    for key, rs in by.items():
        ok = [r for r in rs if not r.get("oom")]
        entry: dict[str, Any] = {
            k: rs[0][k] for k in ("model", "cache", "batch_size", "num_layers", "num_heads",
                                  "num_kv_heads", "hidden_size", "weight_bytes", "attn_implementation")
        }
        entry["passes"] = len(rs)
        entry["oom_passes"] = len(rs) - len(ok)
        for k in SUMMARY_KEYS:
            vals = [r["derived"].get(k) for r in ok if r["derived"].get(k) is not None]
            if not vals:
                continue
            mean = statistics.mean(vals)
            entry[k] = {
                "mean": mean,
                "per_pass": vals,
                "spread_pct": (max(vals) - min(vals)) / mean * 100 if mean else None,
            }
        if ok and ok[0]["profile"].get("ok"):
            entry["top_kernels_pass1"] = ok[0]["profile"]["top_kernels"]
            entry["layer0_kernel_sequence_pass1"] = ok[0]["profile"]["layer0_kernel_sequence"]
            entry["layer0_top_ops_pass1"] = ok[0]["profile"]["layer0_top_ops"]
            entry["host_sync_sites_pass1"] = ok[0]["host_syncs"]["sites"]
        cells[key] = entry
    return cells


def print_table(cells: dict[str, Any]) -> None:
    cols = [("wall_step_ms", "wall"), ("dispatch_step_ms", "disp"), ("gpu_busy_step_ms", "busy"),
            ("gpu_idle_fraction_vs_unprofiled_wall", "idle"), ("kernels_per_step", "k/step"),
            ("kernels_per_layer", "k/layer"), ("gpu_us_per_layer", "gpu_us/L"),
            ("weight_bw_GBps", "wBW"), ("weight_plus_kv_bw_GBps", "w+kvBW"),
            ("generate_ms_per_token", "gen/tok")]
    print("cell".ljust(28) + "".join(h.rjust(10) for _, h in cols))
    for key, c in cells.items():
        row = key.ljust(28)
        for k, _ in cols:
            v = c.get(k, {}).get("mean") if isinstance(c.get(k), dict) else None
            row += (f"{v:10.3f}" if v is not None else "       n/a")
        print(row)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", default=DEFAULT_CELLS,
                   help="'model:cache:b1,b2;...' with cache in {dynamic, static}")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--attention", default="sdpa")
    p.add_argument("--prompt-length", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=8)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--profile-steps", type=int, default=10)
    p.add_argument("--profile-skip", type=int, default=2)
    p.add_argument("--generate-warmup", type=int, default=1)
    p.add_argument("--generate-iters", type=int, default=2)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--rated-bw", type=float, default=672e9, help="rated DRAM bandwidth, bytes/s")
    p.add_argument("--save-traces", action="store_true", help="save gzipped traces of pass 1")
    p.add_argument("--trace-batch-sizes", type=lambda s: [int(x) for x in s.split(",")], default=[1],
                   help="batch sizes whose traces are saved (comma separated)")
    p.add_argument("--trace-budget-mb", type=float, default=20.0)
    p.add_argument("--out-dir", type=Path, default=Path("results/profile_decode"))
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA device required")
    cells = parse_cells(args.cells)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    t_start = time.perf_counter()
    records = []
    for pass_idx in range(args.passes):
        order = cells if pass_idx % 2 == 0 else list(reversed(cells))
        for name, cache, batch in order:
            trace_out = None
            if args.save_traces and pass_idx == 0 and batch in args.trace_batch_sizes:
                trace_out = args.out_dir / "traces" / f"{name.split('/')[-1]}_{cache}_b{batch}.trace.json.gz"
            t0 = time.perf_counter()
            rec = profile_cell(name, cache, batch, args, trace_out)
            rec["pass"] = pass_idx + 1
            records.append(rec)
            d = rec.get("derived", {})
            print(f"[pass {pass_idx + 1}] {cell_key(rec)}: oom={rec.get('oom')} "
                  f"wall={d.get('wall_step_ms', float('nan')):.2f}ms "
                  f"disp={d.get('dispatch_step_ms', float('nan')):.2f}ms "
                  f"busy={d.get('gpu_busy_step_ms', float('nan')):.2f}ms "
                  f"k/step={d.get('kernels_per_step', float('nan'))} "
                  f"gen/tok={d.get('generate_ms_per_token', float('nan')):.2f}ms "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            # write incrementally so a crash keeps completed cells
            (args.out_dir / "raw.json").write_text(
                json.dumps({"environment": env, "config": config, "records": records}, indent=1),
                encoding="utf-8")

    trace_note = None
    tdir = args.out_dir / "traces"
    if tdir.exists():
        total = sum(f.stat().st_size for f in tdir.glob("*.gz")) / 2**20
        if total > args.trace_budget_mb:
            for f in tdir.glob("*.gz"):
                f.unlink()
            tdir.rmdir()
            trace_note = f"traces removed: {total:.1f} MB exceeds budget of {args.trace_budget_mb} MB"
        else:
            trace_note = f"traces kept: {total:.1f} MB"
    summary = {
        "environment": env,
        "config": config,
        "total_seconds": time.perf_counter() - t_start,
        "trace_note": trace_note,
        "cells": summarize(records),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_table(summary["cells"])
    print(f"total {summary['total_seconds']:.0f}s; {trace_note}")


if __name__ == "__main__":
    main()
