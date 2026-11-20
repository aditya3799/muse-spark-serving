"""Isolated, single-GPU full local MoE benchmark. Never edits serving sources."""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace

from tile_study import BLOCKS, PATTERNS, POLICIES, choose_block, make_routes, manifest

SOURCE = "python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py"
EXPECTED_SHA = "18438f083871ec322bdc409690f8b2674cef5381e1e786176eae685dddea2c0f"
BENCH_POLICIES = POLICIES + (
    "sorted_baseline",
    "sorted_ep_density",
    "adaptive_16",
    "adaptive_compact_16",
    "adaptive_compact_32",
)


def load_method(root, policy, torch_module=None):
    sorted_ep = policy.startswith("sorted_")
    tile_policy = policy.removeprefix("sorted_")
    adaptive = tile_policy in (
        "adaptive_16",
        "adaptive_compact_16",
        "adaptive_compact_32",
    )
    if adaptive:
        tile_policy = "baseline"
    path = Path(root) / SOURCE
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA:
        raise ValueError(
            "Source differs from the frozen submitted candidate; audit before running"
        )
    tree = ast.parse(raw.decode("utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MuseSpark1xFp4Experts"
    )
    original = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_marlin_expert_outputs"
    )
    fn = copy.deepcopy(original)
    loops = [
        n
        for n in fn.body
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id == "block_size_m"
    ]
    if len(loops) != 1:
        raise ValueError("Expected exactly one tile selector")
    expected = ast.parse(
        "for block_size_m in _MARLIN_BLOCK_SIZE_M_CANDIDATES:\n"
        "    if num_tokens * topk / self.num_local_experts / block_size_m < 0.9:\n"
        "        break\n"
    ).body[0]
    if ast.dump(loops[0]) != ast.dump(expected):
        raise ValueError("Unexpected baseline tile heuristic")
    if tile_policy != "baseline":
        replacement = ast.parse(
            "block_size_m = choose_block(num_tokens, topk, self.num_local_experts, "
            "self.global_num_experts, EXPERIMENT_POLICY)"
        ).body[0]
        fn.body[fn.body.index(loops[0])] = replacement
    if sorted_ep:
        gemms = [
            node
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "moe_wna16_marlin_gemm"
        ]
        if len(gemms) != 2:
            raise ValueError("Expected exactly two expert GEMMs for sorted-EP opt-in")
        for call in gemms:
            call.keywords.append(
                ast.keyword(
                    arg="allow_sorted_ep",
                    value=ast.parse("expert_map is not None", mode="eval").body,
                )
            )
    namespace = {
        "torch": torch_module,
        "_MARLIN_BLOCK_SIZE_M_CANDIDATES": BLOCKS,
        "choose_block": choose_block,
        "EXPERIMENT_POLICY": tile_policy,
    }
    if adaptive:
        workspace_index = next(
            i
            for i, node in enumerate(fn.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "workspace" for t in node.targets
            )
        )
        fn.body.insert(
            workspace_index,
            ast.parse(
                "adaptive_tables = prepare_adaptive_alignment(local_topk_ids, block_size_m, self.num_local_experts, num_tokens_post_padded, expert_map is not None, sorted_token_ids)"
            ).body[0],
        )
        if policy == "adaptive_compact_32":
            fn.body[workspace_index].value.keywords.append(
                ast.keyword(arg="small_block", value=ast.Constant(32))
            )
        gemms = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "moe_wna16_marlin_gemm"
        ]
        assert len(gemms) == 2
        for call in gemms:
            call.func.id = "adaptive_gemm"
            call.keywords.append(
                ast.keyword(
                    arg="adaptive_tables",
                    value=ast.Name(id="adaptive_tables", ctx=ast.Load()),
                )
            )
            if policy == "adaptive_compact_32":
                call.keywords.append(
                    ast.keyword(arg="adaptive_block", value=ast.Constant(32))
                )
        if torch_module is not None:
            from adaptive_tiles import adaptive_gemm, prepare_adaptive_alignment

            if policy.startswith("adaptive_compact_"):
                from adaptive_tiles import (
                    prepare_compact_alignment as prepare_adaptive_alignment,
                )

            namespace.update(
                adaptive_gemm=adaptive_gemm,
                prepare_adaptive_alignment=prepare_adaptive_alignment,
            )
    body = ast.parse("from __future__ import annotations").body + [fn]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_marlin_expert_outputs"], ast.unparse(fn)


def preflight(root):
    report = {
        "platform": platform.platform(),
        "python": sys.version,
        "source_sha256": EXPECTED_SHA,
        "scope": "one local expert shard; no all-reduce, scheduler, or full model",
    }
    load_method(root, "baseline")
    sys.path.insert(0, str(Path(root).resolve() / "python"))
    import torch

    report.update(torch=torch.__version__, cuda_runtime=torch.version.cuda)
    if not torch.cuda.is_available():
        return dict(report, status="unsupported", reason="CUDA unavailable")
    capability = torch.cuda.get_device_capability()
    report.update(
        gpu=torch.cuda.get_device_name(),
        capability=capability,
        total_memory=torch.cuda.get_device_properties(0).total_memory,
    )
    if platform.system() != "Linux":
        return dict(
            report,
            status="unsupported",
            reason="Pinned SGLang JIT harness requires Linux",
        )
    if not (8 <= capability[0] < 10) or not torch.cuda.is_bf16_supported():
        return dict(
            report,
            status="unsupported",
            reason="This experiment targets SM80-SM90 BF16 Marlin; T4 cannot run it",
        )
    # Check genuine imports before allocating weights; do not substitute kernels.
    from sglang.kernels.ops.moe import moe_wna16_marlin
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_nvfp4_layer_for_marlin,
    )
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    del (
        fused_post_expert_norm_weighted_sum,
        prepare_moe_nvfp4_layer_for_marlin,
        silu_and_mul,
        moe_align_block_size,
    )
    origin = Path(moe_wna16_marlin.__file__).resolve()
    if not origin.is_relative_to(Path(root).resolve()):
        raise RuntimeError("Imported SGLang does not belong to the requested checkout")
    report.update(
        status="ready",
        marlin_origin=str(origin),
        target_h200="H200" in report["gpu"],
        jit_execution_verified=False,
    )
    return report


def make_layer(args, torch):
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_nvfp4_layer_for_marlin,
    )

    # Reuse the checkout's exact NVFP4 test generator without importing the entire test suite.
    path = args.sglang_root / "python/sglang/test/test_marlin_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "make_nvfp4_weight_and_ref"
    )
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    generate = ns[fn.name]
    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=16)
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.params_dtype = torch.bfloat16
    layer.intermediate_size_per_partition = args.intermediate
    layer.num_local_experts = args.local_experts
    layer.global_num_experts = args.global_experts
    mapping = None
    if args.local_experts != args.global_experts:
        mapping = torch.full(
            (args.global_experts,), -1, device="cuda", dtype=torch.int32
        )
        start = args.rank * args.local_experts
        mapping[start : start + args.local_experts] = torch.arange(
            args.local_experts, device="cuda", dtype=torch.int32
        )
    layer.register_buffer("_expert_map", mapping)
    layer.register_buffer(
        "_expert_lookup",
        None if mapping is None else torch.cat((mapping.new_full((1,), -1), mapping)),
    )
    for prefix, n, k in (
        ("w13", 2 * args.intermediate, args.hidden),
        ("w2", args.hidden, args.intermediate),
    ):
        packed, scales, global_scales, reference_weights = [], [], [], []
        for _ in range(args.local_experts):
            w, scale, global_scale, ref = generate(n, k, torch.bfloat16)
            packed.append(w)
            scales.append(scale)
            global_scales.append(global_scale)
            if getattr(args, "retain_reference", False):
                reference_weights.append(ref)
            del ref
        for suffix, values in (
            ("weight", packed),
            ("weight_scale", scales),
            ("weight_scale_2", global_scales),
        ):
            setattr(
                layer,
                prefix + "_" + suffix,
                torch.nn.Parameter(torch.stack(values), requires_grad=False),
            )
        if reference_weights:
            setattr(layer, "reference_" + prefix, torch.stack(reference_weights))
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def comparison(torch, actual, expected, rtol, atol):
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("Nonfinite complete MoE output")
    difference = (actual.float() - expected.float()).abs()
    result = {
        "max_abs": difference.max().item(),
        "relative_l2": (
            difference.norm() / expected.float().norm().clamp_min(1e-20)
        ).item(),
        "bitwise_equal": torch.equal(actual, expected),
        "rtol": rtol,
        "atol": atol,
    }
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return result


def capture(torch, call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    return graph, output


def event_sample(torch, call, repetitions):
    start, stop = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repetitions):
        call()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / repetitions


def run_benchmark(args, report):
    import torch
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import (
        fused_post_expert_norm_weighted_sum,
    )

    if any(p.startswith("sorted_") for p in args.policies):
        from sorted_ep_experiment import install_dispatcher

        report["sorted_ep_experiment"] = install_dispatcher(
            args.sglang_root, args.out / "sorted_ep_overlay"
        )
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    layer = make_layer(args, torch)
    report["weight_preparation_peak_bytes"] = torch.cuda.max_memory_allocated()
    weight = torch.linspace(0.5, 1.5, args.hidden, device="cuda", dtype=torch.bfloat16)
    methods = {
        p: MethodType(load_method(args.sglang_root, p, torch)[0], layer)
        for p in args.policies
    }
    baseline = MethodType(load_method(args.sglang_root, "baseline", torch)[0], layer)
    rows, checks = [], []
    ep = args.global_experts // args.local_experts
    for tokens in args.tokens:
        x = torch.randn((tokens, args.hidden), device="cuda", dtype=torch.bfloat16) / 20
        ids = torch.zeros((tokens, args.topk), device="cuda", dtype=torch.int32)
        scores = torch.rand((tokens, args.topk), device="cuda", dtype=torch.float32)
        scores /= scores.sum(-1, keepdim=True)

        def complete(method):
            values, local_ids = method(x, scores, ids)
            return fused_post_expert_norm_weighted_sum(
                values,
                scores,
                norm_weight=weight,
                eps=1e-6,
                weight_first=True,
                slot_expert_ids=local_ids,
            )

        for policy, method in methods.items():
            # Identical activations/scores across policies, not just within each A/B pair.
            torch.manual_seed(args.seed + tokens)
            # Warm compilation before capture and all measurements.
            ids.copy_(
                torch.tensor(
                    make_routes(
                        tokens, args.topk, args.global_experts, ep, "uniform", args.seed
                    ),
                    device="cuda",
                    dtype=torch.int32,
                )
            )

            def a():
                return complete(baseline)

            def b():
                return complete(method)

            comparison(torch, b(), a(), args.rtol, args.atol)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            graph_a, output_a = capture(torch, a)
            baseline_capture_peak = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            graph_b, output_b = capture(torch, b)
            pair_capture_peak = torch.cuda.max_memory_allocated()
            # Changed-input replay: same pointers, changed routes/activations/scores.
            # Includes zero local work and concentrated routes after live captures.
            for seed in range(2):
                for pattern in args.patterns:
                    report["validation_context"] = dict(
                        tokens=tokens, policy=policy, seed=seed, pattern=pattern
                    )
                    raw = make_routes(
                        tokens,
                        args.topk,
                        args.global_experts,
                        ep,
                        pattern,
                        args.seed + seed,
                    )
                    ids.copy_(torch.tensor(raw, device="cuda", dtype=torch.int32))
                    x.normal_(0, 0.05)
                    scores.uniform_(0.01, 1.0)
                    scores.div_(scores.sum(-1, keepdim=True))
                    expected = a()
                    eager_b = b()
                    check = comparison(torch, eager_b, expected, args.rtol, args.atol)
                    graph_a.replay()
                    replay_a = comparison(
                        torch, output_a, expected, 0, args.replay_atol
                    )
                    graph_b.replay()
                    replay_b = comparison(torch, output_b, eager_b, 0, args.replay_atol)
                    if (
                        max(replay_a["relative_l2"], replay_b["relative_l2"])
                        > args.replay_relative_l2
                    ):
                        raise AssertionError(
                            "Replay exceeds the baseline repeatability L2 screen"
                        )
                    if pattern == "inactive" or (
                        pattern == "all_remote" and args.rank == 0 and ep > 1
                    ):
                        assert torch.count_nonzero(output_b).item() == 0
                    checks.append(
                        dict(
                            tokens=tokens,
                            policy=policy,
                            pattern=pattern,
                            seed=seed,
                            baseline_replay=replay_a,
                            candidate_replay=replay_b,
                            **check,
                        )
                    )
            # Bench each route distribution separately. Record samples, not just a winner.
            for pattern in args.patterns:
                ids.copy_(
                    torch.tensor(
                        make_routes(
                            tokens,
                            args.topk,
                            args.global_experts,
                            ep,
                            pattern,
                            args.seed,
                        ),
                        device="cuda",
                        dtype=torch.int32,
                    )
                )
                for schedule, call_a, call_b in (
                    ("eager", a, b),
                    ("graph", graph_a.replay, graph_b.replay),
                ):
                    for _ in range(3):
                        call_a()
                        call_b()
                    samples = []
                    for pair in range(args.samples):
                        order = (("baseline", call_a), ("candidate", call_b))
                        if pair % 2:
                            order = tuple(reversed(order))
                        sample = {
                            name: event_sample(torch, call, args.repetitions)
                            for name, call in order
                        }
                        sample["order"] = "AB" if pair % 2 == 0 else "BA"
                        samples.append(sample)
                    rows.append(
                        dict(
                            tokens=tokens,
                            policy=policy,
                            pattern=pattern,
                            schedule=schedule,
                            baseline_block=choose_block(
                                tokens,
                                args.topk,
                                args.local_experts,
                                args.global_experts,
                                "baseline",
                            ),
                            candidate_block=choose_block(
                                tokens,
                                args.topk,
                                args.local_experts,
                                args.global_experts,
                                "baseline"
                                if policy.startswith("adaptive_")
                                else policy.removeprefix("sorted_"),
                            ),
                            samples_us=samples,
                            median_paired_ratio=statistics.median(
                                s["baseline"] / s["candidate"] for s in samples
                            ),
                            baseline_capture_peak_bytes=baseline_capture_peak,
                            pair_capture_peak_bytes=pair_capture_peak,
                        )
                    )
            # Bound replay methods would otherwise keep the old graphs alive.
            del call_a, call_b, order
            del graph_a, graph_b, output_a, output_b, expected, eager_b
            gc.collect()
            torch.cuda.empty_cache()
            report.update(timing_rows=rows, checks=checks, status="running")
            (args.out / "benchmark.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(f"Finished M={tokens}, {policy}", flush=True)
    report.update(
        status="passed",
        jit_execution_verified=True,
        timing_rows=rows,
        checks=checks,
        full_model_validated=False,
        recommended_serving_policy=None,
        limitations=[
            "Synthetic weights and routes; single GPU/local shard only.",
            "Tolerance-based differential screen is not greedy-token validation.",
            "Paired capture peaks include two graphs; not deployment memory overhead.",
            "Eager event intervals include launch gaps; graph timings are separate.",
            "No cross-rank critical-path or full-serving speedup measured.",
        ],
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "bench"), default="preflight")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--global-experts", type=int, default=256)
    parser.add_argument("--local-experts", type=int, default=32)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 9, 33, 128])
    parser.add_argument(
        "--policies", nargs="+", choices=BENCH_POLICIES, default=list(POLICIES)
    )
    parser.add_argument(
        "--patterns", nargs="+", choices=PATTERNS, default=list(PATTERNS)
    )
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--replay-atol", type=float, default=0.0)
    parser.add_argument("--replay-relative-l2", type=float, default=0.0)
    args = parser.parse_args()
    if (
        min(
            args.hidden,
            args.intermediate,
            args.local_experts,
            args.global_experts,
            args.topk,
            args.samples,
            args.repetitions,
            *args.tokens,
        )
        <= 0
    ):
        parser.error("Sizes and repeat counts must be positive")
    if args.hidden % 128 or args.intermediate % 128:
        parser.error("Harness dimensions must be multiples of 128")
    if (
        args.global_experts % args.local_experts
        or not 0 <= args.rank < args.global_experts // args.local_experts
        or args.topk > args.global_experts
    ):
        parser.error("Invalid expert partition")
    if args.rtol < 0 or args.atol < 0:
        parser.error("Tolerances must be nonnegative")
    if args.mode == "bench" and args.samples < 2:
        parser.error("At least two samples are needed for alternating AB/BA order")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "starting",
        "started_unix": time.time(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    code = 0
    try:
        report.update(preflight(args.sglang_root))
        if args.mode == "bench" and report["status"] == "ready":
            import torch

            with torch.inference_mode():
                report = run_benchmark(args, report)
        elif args.mode == "bench":
            code = 2
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
        code = 1
    finally:
        report["finished_unix"] = time.time()
        (args.out / "benchmark.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        manifest(args.out)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("checks", "timing_rows")},
                indent=2,
            )
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
