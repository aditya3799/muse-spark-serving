"""Joint producer-to-consumer timing with native pinned target and draft routines."""

import argparse
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess
import threading
import types

import torch
import torch.nn.functional as F

from benchmark import SHA
import kv_bench
from kv_integrated_kernel import _fused_norm_rope_kernel_stacked as fused_direct
from kv_kernels import _fused_norm_rope_kernel_stacked, set_kv_buffer_prefix_valid_tiled

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research_2026-11-18_rl"))
import bench_aux
from build_aux import PARENT, MODEL, change

WIDTH, LAYERS, DIM, HIDDEN = 8, 5, 128, 8192
POOL_ROWS = 32768
ARMS = ("baseline", "combined", "control", "direct_kv", "packed_taps")


def native_rms():
    bridge = ctypes.CDLL(str(HERE / "rms_flashinfer_072.dll"))
    entry = bridge.run_bf16_rms
    ptr = ctypes.c_void_p
    entry.argtypes = [ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_float, ptr]
    entry.restype = ctypes.c_int

    def call(x, weight, out):
        status = entry(
            x.data_ptr(),
            weight.data_ptr(),
            out.data_ptr(),
            x.shape[0],
            x.shape[1],
            1e-5,
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError("native FlashInfer RMSNorm code " + str(status))

    return call


def validated_consumer():
    # Compile the exact functions already used by the paired direct-KV A/B.
    r = ast.parse((HERE / "kv_bench.py").read_text(encoding="utf-8-sig"))
    case = next(n for n in r.body if getattr(n, "name", None) == "run_case")
    call = next(n for n in case.body if getattr(n, "name", None) == "call")
    return compile(ast.Module(body=[call], type_ignores=[]), "kv_bench.py", "exec")


def scoped_functions():
    source = (PARENT / "candidate_sources" / MODEL).read_text()
    return (
        bench_aux.forward(source),
        bench_aux.forward(change(source)),
        bench_aux.pack_function(),
    )


def model_stub(snapshots):
    return types.SimpleNamespace(
        embed_norm=None,
        residual_dtype=torch.float32,
        operator_dtype=torch.bfloat16,
        layers=[bench_aux.ResidualLayer(x) for x in snapshots],
        norm=lambda x: x,
        layers_to_capture=[0, 1, 2, 3, 4],
    )


def cpu():
    assert hashlib.sha256((PARENT / "candidate.diff").read_bytes()).hexdigest() == SHA
    norm = (
        native_rms()
    )  # Loading is CPU-only, and checks that the runtime DLL resolves.
    assert callable(norm)
    a, z, pack = scoped_functions()
    g = torch.Generator().manual_seed(173301)
    snapshots = [torch.randn(17, 64, generator=g) for _ in range(5)]
    model = model_stub(snapshots)
    batch = types.SimpleNamespace(forward_mode=None)
    embed = snapshots[0].bfloat16()
    xb = bench_aux.call(a, model, batch, embed, pack)[1]
    xc = bench_aux.call(z, model, batch, embed, pack)[1]
    assert torch.equal(xb, xc)
    w = torch.randn(64, 320, generator=g).bfloat16()
    for input_tensor in (xb, xc):
        assert torch.equal(F.linear(input_tensor, w), F.linear(xb, w))
    validated_consumer()
    print(
        "Original scored producer capture/pack, native RMSNorm DLL, full-K geometry and pinned consumer imports ready."
    )


def run_case(b, seed, fc, kv_weight, norm_weight, rope, rms, rounds):
    gen = torch.Generator(device="cuda").manual_seed(seed + b * 193)
    m = b * WIDTH

    def rand(shape, dtype):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=gen)

    snapshots = [rand((m, HIDDEN), torch.float32) for _ in range(LAYERS)]
    embed = snapshots[0].bfloat16()
    model = model_stub(snapshots)
    batch = types.SimpleNamespace(forward_mode=None)
    source_a, source_b, pack = scoped_functions()

    def capture(fn):
        bench_aux.call(fn, model, batch, embed, pack)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = bench_aux.call(fn, model, batch, embed, pack)
        return graph, outputs

    producer = {"original": capture(source_a), "packed": capture(source_b)}
    norm = rand((LAYERS, DIM), torch.bfloat16)
    eps = torch.full((LAYERS,), 1e-5, device="cuda")
    loc = torch.arange(m, device="cuda", dtype=torch.int64) + 1
    pos = torch.zeros_like(loc)
    lens = torch.full((b,), 4, device="cuda", dtype=torch.int32)
    temp_k = rand((LAYERS, m, 1, DIM), torch.bfloat16)
    temp_v = torch.empty_like(temp_k)
    native = validated_consumer()
    streams = {}
    for name in ARMS:
        keys = [
            torch.zeros((POOL_ROWS + 1, 1, DIM), device="cuda", dtype=torch.bfloat16)
            for _ in range(LAYERS)
        ]
        values = [torch.zeros_like(t) for t in keys]
        pool = (keys, values, kv_bench.pointers(keys), kv_bench.pointers(values))
        output = rand((m, HIDDEN), torch.bfloat16)
        projected = rand((m, LAYERS, 2 * DIM), torch.bfloat16)
        ns = dict(
            torch=torch,
            h=output,
            weight=kv_weight,
            proj=projected,
            norm=norm,
            eps=eps,
            rope=rope,
            pos=pos,
            temp_k=temp_k,
            temp_v=temp_v,
            loc=loc,
            lens=lens,
            b=b,
            m=m,
            WIDTH=WIDTH,
            DIM=DIM,
            LAYERS=LAYERS,
            _fused_norm_rope_kernel_stacked=_fused_norm_rope_kernel_stacked,
            fused_direct=fused_direct,
            set_kv_buffer_prefix_valid_tiled=set_kv_buffer_prefix_valid_tiled,
        )
        exec(native, ns)
        streams[name] = (pool, output, projected, ns["call"])
    modes = {
        name: (
            "packed" if name in ("packed_taps", "combined") else "original",
            name in ("direct_kv", "combined"),
        )
        for name in ARMS
    }

    def launch(name):
        source_mode, direct = modes[name]
        graph, outputs = producer[source_mode]
        graph.replay()
        # Preserve the scored model's actual replicated BF16 full-K projection.
        context = F.linear(outputs[1], fc)
        pool, normalized, _, consumer = streams[name]
        rms(context, norm_weight, normalized)
        consumer(direct, pool)

    for name in ARMS:
        launch(name)
    torch.cuda.synchronize()
    # Independent upstream native vs pinned PyTorch RMS math check and error.
    x = F.linear(producer["original"][1][1], fc)
    oracle = (
        (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5))
        * norm_weight
    ).bfloat16()
    nr = streams["baseline"][1]
    rms_delta = (nr.float() - oracle.float()).abs()
    rms_info = dict(
        max_abs=rms_delta.max().item(),
        bitwise_fraction=(nr == oracle).float().mean().item(),
    )
    assert torch.allclose(nr, oracle, rtol=0.008, atol=0.001)
    gpu = {k: [] for k in ARMS}
    wall = {k: [] for k in ARMS}
    order_log = []
    tested = 0
    wrong = {k: 0 for k in ARMS[1:]}
    for step in range(rounds):
        for tensor in snapshots:
            tensor.normal_(generator=gen)
        loc.copy_(torch.randperm(POOL_ROWS, device="cuda", generator=gen)[:m] + 1)
        pos.random_(40000, 100000, generator=gen)
        if step % 8 == 0:
            lens.fill_(8)
        else:
            lens.random_(1, 9, generator=gen)
        # Cyclic Latin square and inverse: each arm occupies every order slot.
        order = list(ARMS[step % 5 :] + ARMS[: step % 5])
        if (step // 5) % 2:
            order.reverse()
        order_log.append(order)
        for name in order:
            a, z = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            torch.cuda.synchronize()
            t = time.perf_counter_ns()
            a.record()
            launch(name)
            z.record()
            z.synchronize()
            wall[name].append((time.perf_counter_ns() - t) / 1000)
            gpu[name].append(a.elapsed_time(z) * 1000)
        assert torch.equal(producer["packed"][1][1], producer["original"][1][1])
        for name in ARMS[1:]:
            assert torch.equal(streams[name][1], streams["baseline"][1])
            assert torch.equal(streams[name][2], streams["baseline"][2])
        # Exact BF16 bits for every active and rejected physical row, with a
        # complete final pass over each backing pool to catch stray writes.
        ix = loc if step + 1 < rounds else slice(None)
        for part in (0, 1):
            for l in range(LAYERS):
                base = streams["baseline"][0][part][l][ix].view(torch.int16)
                for name in ARMS[1:]:
                    val = streams[name][0][part][l][ix].view(torch.int16)
                    wrong[name] += (base != val).sum().item()
                tested += base.numel()
    return dict(
        wall_us=wall,
        cuda_event_us=gpu,
        orders=order_log,
        cache_values_per_arm=tested,
        wrong=wrong,
        upstream_cuda_rms_vs_pinned_native_math=rms_info,
    )


def monitor_gpu(stop, samples):
    while not stop.is_set():
        try:
            raw = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=clocks.current.graphics,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                .strip()
                .splitlines()[0]
            )
            vals = [
                float(x.strip())
                if x.strip() not in ("N/A", "[N/A]", "[Not Supported]")
                else None
                for x in raw.split(",")
            ]
            samples.append(dict(graphics_mhz=vals[0], watts=vals[1], temp_c=vals[2]))
        except (ValueError, subprocess.SubprocessError, IndexError, OSError):
            pass
        stop.wait(0.4)


def main(stage, rounds):
    if stage == "cpu":
        return cpu()
    assert rounds % 10 == 0
    assert hashlib.sha256((PARENT / "candidate.diff").read_bytes()).hexdigest() == SHA
    seed = 390413 if stage == "pilot" else 954107
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def weight(shape, scale):
        t = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        return t.normal_(std=scale, generator=gen)

    fc = weight((HIDDEN, 5 * HIDDEN), (5 * HIDDEN) ** -0.5)
    kv = weight((HIDDEN, 2 * LAYERS * DIM), HIDDEN**-0.5)
    norm_w = weight((HIDDEN,), 1.0)
    rope = weight((131072, DIM), 1.0)
    rms = native_rms()
    result = dict(seed=seed, stage=stage, details={}, complete=False)
    for b in (1, 4, 16, 32):
        observations = []
        stopped = threading.Event()
        telemetry = threading.Thread(
            target=monitor_gpu, args=(stopped, observations), daemon=True
        )
        telemetry.start()
        try:
            result["details"][f"b{b}"] = run_case(
                b, seed, fc, kv, norm_w, rope, rms, rounds
            )
        finally:
            stopped.set()
            telemetry.join(timeout=3)
        result["details"][f"b{b}"]["telemetry"] = observations
        torch.cuda.empty_cache()
        print(stage, b, result["details"][f"b{b}"]["wrong"], flush=True)
    result["complete"] = True
    (HERE / f"joint_{stage}_detail.json").write_text(json.dumps(result, indent=2))
    export_canonical(result)


def export_canonical(result):
    stage = result["stage"]
    seed = result["seed"]
    rounds = len(next(iter(result["details"].values()))["orders"])
    indices = [
        base + offset
        for base in range(0, rounds, 10)
        for i in range(5)
        for offset in (i, i + 5)
    ]
    for state in result["details"].values():
        assert all(
            state["orders"][i] == list(reversed(state["orders"][j]))
            for i, j in zip(indices[::2], indices[1::2])
        )
    for arm in ["combined", "direct_kv", "packed_taps"]:
        canonical = dict(
            agent="speculation",
            candidate="joint_" + arm,
            scope="single SM86 CUDA-graph target-tap packing, replicated BF16 full-K fc, FlashInfer 0.7.2 native BF16 RMSNorm, native dense draft-KV projection and fused norm/RoPE/cache writes; fixed five upstream residual snapshots, no target/draft layers, TP or model weights",
            baseline_sha256=SHA,
            correct=all(
                v["wrong"][arm] == 0 and v["wrong"]["control"] == 0
                for v in result["details"].values()
            ),
            active=True,
            stage=stage,
            seed=seed,
            unit="us synchronized wall",
            order_indices=indices,
            pairing="Reciprocal within each 10-step cyclic Latin square and inverse, raw chronological order kept in detail",
            bench={"instances": 1, "shapes": {}},
            complete=True,
        )
        for name, state in result["details"].items():
            t = {
                k: [samples[i] for i in indices]
                for k, samples in state["wall_us"].items()
            }
            canonical["bench"]["shapes"][name] = dict(
                weight=1,
                times={
                    "baseline": t["baseline"],
                    "control": t["control"],
                    "candidate": t[arm],
                },
            )
        (HERE / f"joint_{arm}_{stage}.json").write_text(json.dumps(canonical, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("cpu", "pilot", "heldout"))
    p.add_argument("--rounds", type=int, default=20)
    args = p.parse_args()
    with torch.inference_mode():
        main(args.stage, args.rounds)
