"""Step-one GPU preflight. No package installs, network calls, or model weights."""
import csv
import hashlib
import html
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def route_oracle(ids, local_start=64, local_count=32, block=8):
    """Independent CPU specification, including the inactive (-1) bucket."""
    local = [x - local_start if local_start <= x < local_start + local_count else -1 for x in ids]
    counts = [local.count(e) for e in range(-1, local_count)]
    padded = [math.ceil(c / block) * block for c in counts]
    ordered = sorted(range(len(local)), key=lambda i: local[i])
    packed, experts = [], []
    for expert in range(-1, local_count):
        slots = [i for i, e in enumerate(local) if e == expert]
        n = math.ceil(len(slots) / block) * block
        packed.extend(slots + [len(local)] * (n - len(slots)))
        experts.extend([expert] * (n // block))
    return dict(local=local, counts=counts, padded=padded, ordered=ordered,
                packed=packed, experts=experts)


def make_routes(tokens, mode, seed=604):
    """Eight unique global experts per token; optional explicit drop sentinel."""
    rng = random.Random(seed)
    ids = []
    remote = [e for e in range(256) if not 64 <= e < 96]
    for _ in range(tokens):
        if mode == "all_local":
            row = rng.sample(range(64, 96), 8)
        elif mode == "all_remote":
            row = rng.sample(remote, 8)
        elif mode == "hot_local":
            hot = [64, 65, 66, 67]
            row = hot + rng.sample([e for e in range(256) if e not in hot], 4)
        else:
            row = rng.sample(range(256), 8)
            if mode == "sentinel":
                row[0] = -1
        ids.extend(row)
    return ids


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Preflight:
    def __init__(self, manifest, source, output_root=None, force_cpu=False):
        self.manifest, self.source = manifest, source
        self.force_cpu = force_cpu
        base = Path(output_root) if output_root else (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd())
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        self.out = base / ("v4_preflight_" + run_id)
        self.out.mkdir(parents=True, exist_ok=False)
        self.report = dict(schema_version=1, run_id=run_id, created_utc=datetime.now(timezone.utc).isoformat(),
                           source_manifest=manifest, sections={},
                           scope="Compatibility and synthetic component diagnostics; NOT a serving benchmark or H200 prediction.")
        self.torch = None
        self.gpu = False
        self.trace_case = None
        self.save()

    def save(self):
        (self.out / "report.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")

    def section(self, name, fn):
        start = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            result = dict(status="failed", error=type(exc).__name__ + ": " + str(exc),
                          traceback=traceback.format_exc(limit=8))
        result["wall_seconds"] = round(time.perf_counter() - start, 3)
        self.report["sections"][name] = result
        self.save()
        print(name + ": " + result.get("status", "unknown"))
        print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "samples", "traceback", "timing_cases")}, indent=2))
        return result

    def environment(self):
        def run():
            versions = {}
            for package in ("torch", "triton", "numpy", "ipykernel"):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = None
            result = dict(status="passed", python=sys.version, os=platform.system(),
                          machine=platform.machine(), packages=versions,
                          tools={x: bool(shutil.which(x)) for x in ("nvidia-smi", "nvcc", "nsys", "ncu")},
                          forced_cpu=self.force_cpu, gpu_devices=[])
            try:
                import torch
                self.torch = torch
                result.update(torch_cuda_version=torch.version.cuda, cuda_available=torch.cuda.is_available())
                self.gpu = bool(torch.cuda.is_available() and not self.force_cpu)
                if self.gpu:
                    torch.cuda.set_device(0)
                    for i in range(torch.cuda.device_count()):
                        p = torch.cuda.get_device_properties(i)
                        result["gpu_devices"].append(dict(index=i, name=p.name,
                            capability=[p.major, p.minor], memory_bytes=p.total_memory,
                            multiprocessors=p.multi_processor_count))
                    result["active_device"] = 0
                    result["native_bf16_expected"] = torch.cuda.get_device_capability()[0] >= 8
                else:
                    result["status"] = "cpu_only"
                    result["note"] = "Select GPU T4 in Kaggle Settings > Accelerator, then restart and Run All. CPU routing anatomy still works."
            except ImportError as exc:
                result.update(status="cpu_only", note="PyTorch is missing; no installation attempted.", error=str(exc))
            if shutil.which("nvidia-smi") and not self.force_cpu:
                try:
                    p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,pstate,power.draw,clocks.sm",
                                        "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
                    result["nvidia_smi_summary"] = p.stdout.strip()
                    result["nvidia_smi_returncode"] = p.returncode
                except Exception as exc:
                    result["nvidia_smi_note"] = str(exc)
            return result
        return self.section("environment", run)

    def measure(self, fn, repeats=20, samples=15):
        """CUDA-event interval and synchronized wall time; neither is pure kernel sum."""
        t = self.torch
        for _ in range(5):
            fn()
        t.cuda.synchronize()
        device_us, wall_us = [], []
        for _ in range(samples):
            a, b = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
            begin = time.perf_counter()
            a.record()
            for _ in range(repeats):
                fn()
            b.record()
            b.synchronize()
            wall_us.append((time.perf_counter() - begin) * 1e6 / repeats)
            device_us.append(a.elapsed_time(b) * 1000 / repeats)
        values = sorted(device_us)
        return dict(event_us_median=statistics.median(values), event_us_p10=values[int(.1 * (len(values) - 1))],
                    event_us_p90=values[int(.9 * (len(values) - 1))], wall_us_median=statistics.median(wall_us),
                    raw_event_us=device_us, raw_wall_us=wall_us, repeats=repeats, samples=samples,
                    method="Warm eager calls, reused inputs, output allocation included where used. CUDA-event intervals include exposed stream gaps. No H200 extrapolation.")

    def timing_and_graphs(self):
        def run():
            if not self.gpu:
                return dict(status="skipped", reason="No active CUDA GPU")
            t = self.torch
            x = t.arange(4096, device="cuda", dtype=t.float32)
            y, out = t.ones_like(x), t.empty_like(x)
            def fn():
                return t.add(x, y, out=out)
            fn()
            t.testing.assert_close(out, x + y, rtol=0, atol=0)
            timing = self.measure(fn)
            stream = t.cuda.Stream()
            stream.wait_stream(t.cuda.current_stream())
            with t.cuda.stream(stream):
                for _ in range(5):
                    fn()
            t.cuda.current_stream().wait_stream(stream)
            graph = t.cuda.CUDAGraph()
            with t.cuda.graph(graph):
                fn()
            for shift in (1, 3, 7):
                x.add_(shift)
                graph.replay()
                t.testing.assert_close(out, x + y, rtol=0, atol=0)
            return dict(status="passed", changing_input_graph_replays=3, eager_timing=timing,
                        graph_timing=self.measure(graph.replay), note="Timing infrastructure test, not a model operation.")
        return self.section("timing_and_graphs", run)

    def dtype_checks(self):
        def run():
            if not self.gpu:
                return dict(status="skipped", reason="No active CUDA GPU")
            t, cases = self.torch, []
            for name, dtype in (("fp32", t.float32), ("fp16", t.float16), ("bf16", t.bfloat16)):
                if name == "bf16" and t.cuda.get_device_capability()[0] < 8:
                    cases.append(dict(dtype=name, status="skipped", reason="No native BF16 Tensor Core path on this architecture; emulation is not substituted."))
                    continue
                try:
                    # Small integer matrices have exactly representable expected sums.
                    a = (t.arange(32 * 64, device="cuda").reshape(32, 64) % 3).to(dtype)
                    b = (t.arange(64 * 16, device="cuda").reshape(64, 16) % 2).to(dtype)
                    expected = a.cpu().double() @ b.cpu().double()
                    actual = a @ b
                    t.testing.assert_close(actual.cpu().double(), expected, rtol=0, atol=0)
                    cases.append(dict(dtype=name, status="passed"))
                except Exception as exc:
                    cases.append(dict(dtype=name, status="failed", error=str(exc)))
            return dict(status="passed" if not any(c["status"] == "failed" for c in cases) else "partial", cases=cases,
                        results=cases, note="Basic dtype operation check only; not model numerical validation.")
        return self.section("dtype_checks", run)

    def original_kernel_probe(self):
        def run():
            evidence = dict(marlin="Not executed: pinned source contains empty device implementations below SM80.",
                            fa3="Not executed: out of scope for this T4 preflight.")
            if not self.gpu:
                return dict(status="skipped", reason="No active CUDA GPU", **evidence)
            t = self.torch
            if t.cuda.get_device_capability()[0] < 8:
                return dict(status="skipped", reason="Current Triton support starts at SM80. Do not downgrade libraries or remove architecture guards for this notebook.", **evidence)
            if importlib.util.find_spec("triton") is None:
                return dict(status="skipped", reason="Triton is not installed; no packages installed automatically.", **evidence)
            expected_hash = self.manifest["embedded_alignment_sha256"]
            assert hashlib.sha256(self.source.encode()).hexdigest() == expected_hash
            dependency = "from sglang.kernels.jit.utils import is_arch_support_pdl"
            assert self.source.count(dependency) == 1
            adapted = self.source.replace(dependency,
                "# Notebook-only launcher adapter: PDL disabled; kernel source unchanged.\ndef is_arch_support_pdl():\n    return False")
            source_path = self.out / "v4_alignment_probe.py"
            source_path.write_text(adapted, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("v4_alignment_probe_" + self.report["run_id"], source_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            verified = 0
            for mode in ("uniform", "all_remote", "all_local", "sentinel"):
                values = make_routes(8, mode)
                oracle = route_oracle(values)
                raw = t.tensor(values, device="cuda", dtype=t.int64).reshape(8, 8)
                mapping = t.full((256,), -1, device="cuda", dtype=t.int32)
                mapping[64:96] = t.arange(32, device="cuda", dtype=t.int32)
                local = t.full_like(raw, -999, dtype=t.int32)
                ordered = t.full((512,), -999, device="cuda", dtype=t.int32)
                experts = t.full((64,), -999, device="cuda", dtype=t.int32)
                count = t.full((1,), -999, device="cuda", dtype=t.int32)
                module.moe_align_small_numel(raw, 33, 8, ordered, experts, count, expert_map=mapping, local_topk_ids=local)
                t.cuda.synchronize()
                n = int(count.item())
                assert local.flatten().tolist() == oracle["local"]
                assert n == len(oracle["packed"])
                assert ordered[:n].tolist() == oracle["packed"]
                assert experts[:n // 8].tolist() == oracle["experts"]
                verified += 1
            return dict(status="passed", correctness_cases=verified,
                        implementation="V4 alignment kernel; notebook-only PDL-disabled import adapter. No GEMM stubs or dtype changes.", **evidence)
        return self.section("original_kernel_probe", run)

    def routing_diagnostic(self):
        def run():
            rows, timings = [], []
            t = self.torch
            for tokens in (1, 8, 9, 32, 128, 256):
                for mode in ("uniform", "hot_local", "all_remote", "all_local", "sentinel"):
                    values = make_routes(tokens, mode)
                    oracle = route_oracle(values)
                    active = sum(c for c in oracle["counts"][1:])
                    rows.append(dict(tokens=tokens, routes=len(values), pattern=mode,
                        live_routes=active, inactive_routes=oracle["counts"][0],
                        active_padded_rows=sum(oracle["padded"][1:]),
                        inactive_padded_rows=oracle["padded"][0],
                        active_padding_overhead_rows=sum(oracle["padded"][1:]) - active,
                        expert_block_size=8, global_experts=256, local_experts=32,
                        original_v4_small_route_eligible=len(values) <= 64))
                    if not self.gpu:
                        continue
                    raw = t.tensor(values, device="cuda", dtype=t.int64)
                    mapping = t.full((256,), -1, device="cuda", dtype=t.int32)
                    mapping[64:96] = t.arange(32, device="cuda", dtype=t.int32)
                    def map_routes():
                        # Same sentinel-preserving expression as the inspected SGLang helper.
                        local_ids = raw.to(t.int32)
                        return t.where(local_ids < 0, -1, mapping[local_ids.clamp_min(0)])
                    def sequence():
                        with t.profiler.record_function("diagnostic/map_global_to_local"):
                            local = map_routes()
                        with t.profiler.record_function("diagnostic/sort_and_count"):
                            order = t.argsort(local, stable=True)
                            counts = t.zeros(33, device=raw.device, dtype=t.int64)
                            counts.scatter_add_(0, local.long() + 1, t.ones_like(local, dtype=t.int64))
                        with t.profiler.record_function("diagnostic/padding_counts"):
                            padded = ((counts + 7) // 8) * 8
                        return local, order, counts, padded
                    outputs = sequence()
                    for got, key in zip(outputs, ("local", "ordered", "counts", "padded")):
                        assert got.tolist() == oracle[key], (tokens, mode, key)
                    # Time a bounded subset after correctness; no claim of stock-vs-V4 speedup.
                    if tokens in (8, 128) and mode in ("uniform", "hot_local"):
                        result = self.measure(sequence, repeats=10, samples=15)
                        timings.append(dict(tokens=tokens, pattern=mode, **result))
                    if tokens == 128 and mode == "hot_local":
                        # Bind tensors in default arguments so later loop iterations cannot change them.
                        self.trace_case = (raw, mapping)
            write_csv(self.out / "routing_structure.csv", rows)
            write_csv(self.out / "routing_timings.csv", [dict(tokens=x["tokens"], pattern=x["pattern"],
                      event_us_median=x["event_us_median"], event_us_p10=x["event_us_p10"],
                      event_us_p90=x["event_us_p90"], wall_us_median=x["wall_us_median"]) for x in timings])
            return dict(status="passed" if self.gpu else "cpu_only", structural_cases=len(rows),
                        gpu_correctness_cases=len(rows) if self.gpu else 0, timing_cases=timings,
                        implementation="PyTorch diagnostic map/sort/histogram/padding-count sequence. It does NOT emit the packed aligned buffer and is NOT the stock or V4 alignment implementation.",
                        limitations="Synthetic routes for one rank; fixed block size 8; eager execution; no expert GEMMs, attention, scheduler, communication, or model weights. Timings are not serving fractions.")
        return self.section("routing_diagnostic", run)

    def profile(self):
        def run():
            if not self.gpu or self.trace_case is None:
                return dict(status="skipped", reason="No validated CUDA routing diagnostic")
            t = self.torch
            raw, mapping = self.trace_case
            def step():
                with t.profiler.record_function("diagnostic/map_global_to_local"):
                    ids = raw.to(t.int32)
                    local = t.where(ids < 0, -1, mapping[ids.clamp_min(0)])
                with t.profiler.record_function("diagnostic/sort_and_count"):
                    order = t.argsort(local, stable=True)
                    counts = t.zeros(33, device=raw.device, dtype=t.int64)
                    counts.scatter_add_(0, local.long() + 1, t.ones_like(local, dtype=t.int64))
                with t.profiler.record_function("diagnostic/padding_counts"):
                    padded = ((counts + 7) // 8) * 8
                return local, order, counts, padded
            for _ in range(5):
                step()
            t.cuda.synchronize()
            with t.profiler.profile(activities=[t.profiler.ProfilerActivity.CPU, t.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True, with_stack=False) as prof:
                for _ in range(6):
                    step()
                    prof.step()
                t.cuda.synchronize()
            prof.export_chrome_trace(str(self.out / "routing_trace.json"))
            operations = []
            for event in prof.key_averages():
                operations.append(dict(operation=event.key, count=event.count,
                    self_cpu_us=event.self_cpu_time_total,
                    self_device_us=getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0)),
                    self_cpu_memory_bytes=event.self_cpu_memory_usage,
                    self_device_memory_bytes=getattr(event, "self_device_memory_usage", getattr(event, "self_cuda_memory_usage", 0))))
            write_csv(self.out / "profile_operations.csv", operations)
            device_events = [e for e in prof.events() if str(e.device_type).endswith("CUDA")]
            kernels = Counter(e.name for e in device_events)
            write_csv(self.out / "profile_device_events.csv", [dict(name=k, count=v) for k, v in kernels.items()])
            return dict(status="passed" if device_events else "partial", captured_steps=6,
                        device_event_count=len(device_events), cpu_operation_groups=len(operations),
                        gpu_trace_available=bool(device_events),
                        capture_note="CUDA events recorded." if device_events else "No CUDA events recorded; CPU trace was still saved. CUDA-event timings from section 5 are separate measurements.",
                        note="Trace of the synthetic PyTorch diagnostic only. Device events may include copies and kernels. Parent/child CPU ranges overlap; do not sum them into a serving breakdown.",
                        trace="routing_trace.json", nsight="Presence only checked; hardware-counter access not tested.")
        return self.section("profiler", run)

    def export(self):
        self.save()
        statuses = {k: v.get("status") for k, v in self.report["sections"].items()}
        expected = ("environment", "timing_and_graphs", "dtype_checks", "original_kernel_probe", "routing_diagnostic", "profiler")
        missing = [x for x in expected if x not in statuses]
        gpu = self.report["sections"].get("environment", {}).get("gpu_devices", [])
        summary = ["# V4 GPU preflight results", "", "Scope: synthetic compatibility/component diagnostics, not a full serving profile.", "",
                   "Hardware: " + (", ".join(x["name"] for x in gpu) or "CPU-only / no CUDA GPU"), "",
                   "| Section | Status |", "|---|---|"]
        summary += ["| " + k + " | " + v + " |" for k, v in statuses.items()]
        if missing:
            summary += ["", "Missing sections: " + ", ".join(missing)]
        summary += ["", "Interpretation", "- Passed means the stated small test passed, not model correctness.",
                    "- Skipped original Triton/Marlin paths on T4 are expected; do not interpret them as zero-cost operations.",
                    "- routing_structure.csv describes CPU-calculated route/padding work.",
                    "- routing_timings.csv and routing_trace.json describe an adapted PyTorch diagnostic, not stock/V4 alignment.",
                    "- There are no H200 runtime percentages or serving speedup claims.",
                    "- Keep raw samples and software versions when comparing future runs.", "",
                    "Next: review compatibility and select original supported kernels for notebook 02."]
        text = "\n".join(summary) + "\n"
        (self.out / "SUMMARY.txt").write_text(text, encoding="utf-8")
        (self.out / "summary.html").write_text("<!doctype html><meta charset='utf-8'><title>V4 preflight</title><style>body{font:16px system-ui;margin:32px;max-width:1000px}pre{white-space:pre-wrap}</style><pre>" + html.escape(text) + "</pre>", encoding="utf-8")
        members = sorted(p for p in self.out.iterdir() if p.is_file())
        checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in members}
        (self.out / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
        archive = self.out.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():
                    z.write(p, p.name)
        print(text)
        print("Download:", archive)
        return archive
