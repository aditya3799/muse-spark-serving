"""Exercise the actual scored full-K Humming kernels on complementary output features."""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "placement"))
from worker import BASE, cached_threshold_for_preparation, finish, fixture, native, torch
from humming.layer import HummingMethod
from sglang.kernels.ops.moe.muse_spark_humming_split import choose_partition_configs, partition_routes
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.muse_spark_humming_scale import scale_active_rows
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size


def make_weights(experts, width):
    weights = {}
    for stage, n in (("w13", 2 * width), ("w2", width)):
        weights[stage] = (
            torch.randint(256, (experts, n, width // 2), dtype=torch.uint8),
            torch.randint(40, 68, (experts, n, width // 16), dtype=torch.uint8),
            torch.linspace(.125, .48, experts).to(torch.bfloat16),
        )
    return weights


def pack(weights, rank, feature_shard):
    experts, width = weights["w2"][0].shape[:2]
    half = width // 2
    raw = torch.nn.Module()
    for stage, (weight, scale, global_scale) in weights.items():
        if feature_shard:
            features = torch.arange(rank * half, (rank + 1) * half)
            if stage == "w13":
                features = torch.cat((features, width + features))
            weight = weight.index_select(1, features)
            scale = scale.index_select(1, features)
        else:
            local = slice(rank * (experts // 2), (rank + 1) * (experts // 2))
            weight, scale, global_scale = weight[local], scale[local], global_scale[local]
        for suffix, val in (("weight", weight), ("weight_scale", scale.view(torch.float8_e4m3fn)), ("weight_scale_2", global_scale)):
            raw.register_parameter(stage + "_" + suffix, torch.nn.Parameter(val.cuda(), requires_grad=False))
    return native["_prepare_humming_experts"](raw, torch)


def stage(layer, name, x, ids, route_plan=None):
    valid = ids.numel()
    config = next(c for lo, hi, c in layer.fp32_configs[name] if lo < valid <= hi)
    assert not config["use_stream_k"] and not config["use_f16_accum"]
    width = layer.humming_metas[name].shape_n
    output = x.new_empty((valid, width))
    parts=[(ids,{name:config})] if route_plan is None else route_plan
    for part, configs in parts:
        tuning=configs[name]
        sorted_ids, expert_ids, padded = moe_align_block_size(part, tuning["block_shape"][0], layer.w2_post_scale.numel(), ignore_invalid_expert=True)
        HummingMethod.forward_layer(layer, inputs=x, outputs=output, top_k=ids.shape[1] if name == "w13" else 1,
            sublayer_name=name, tuning_config=tuning, sorted_ids=sorted_ids, expert_ids=expert_ids, num_tokens_padded=padded,
            valid_shape_m=valid, compute_config={"gemm_type":"indexed","use_f16_accum": False})
    if name == "w2":
        return output
    scale_active_rows(output, ids, layer.w13_post_scale)
    hidden = output.new_empty((valid, width // 2))
    silu_and_mul(output, hidden, expert_ids=ids.flatten(), expert_step=1)
    return hidden


def graphs(parent, shard, m, h, e, seed, pattern, rounds, preserve_tiles):
    x = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
    ids = torch.empty(m, 8, device="cuda", dtype=torch.int32)
    scores = torch.empty(m, 8, device="cuda")
    norm = torch.linspace(.7, 1.2, h, device="cuda", dtype=x.dtype)
    group_size = e // 2

    def fill(index):
        gen = torch.Generator(device="cuda").manual_seed(seed + index * 271)
        x.normal_(std=.023, generator=gen)
        keys = torch.rand(m, 4*e, device="cuda", generator=gen)
        if pattern == "skewed":
            keys[::2, :max(4, group_size // 2)] += 1.
        chosen = keys.topk(8).indices
        ids.copy_(torch.where(chosen < e, chosen, -1))
        scores.uniform_(generator=gen)
        scores.div_(scores.sum(-1, keepdim=True))

    def run_parent():
        pieces = []
        experts_out = []
        for rank, layer in enumerate(parent):
            local = ids - rank * group_size
            local = torch.where((local >= 0) & (local < group_size), local, -1)
            down, local = native["_humming_expert_outputs"](layer, x, local, torch, global_experts=4*e, defer_downscale=True)
            pieces.append(finish(down, scores, norm_weight=norm, eps=1e-8, weight_first=True,
                                 slot_expert_ids=local, expert_output_scale=layer.w2_post_scale))
            experts_out.append((down, local))
        return pieces[0]+pieces[1], experts_out

    def run_candidate():
        # A real link would carry the BF16 activation shards. Here both ranks run serially.
        route_plan=None
        if preserve_tiles:
            # Each global expert has the same count on both layouts. Match its original
            # light/heavy choice and both complete K/warp geometries before slicing columns.
            if ids.numel()>4096:
                raise ValueError("matched-geometry prototype covers small-route dispatch only")
            plan=choose_partition_configs(parent[0],ids.numel(),4*e)
            if plan is None:
                native_cfg={s:next(c for lo,hi,c in cfg if lo<ids.numel()<=hi) for s,cfg in parent[0].fp32_configs.items()}
                route_plan=[(ids,native_cfg)]
            else:
                light,heavy,_=partition_routes(ids,e,plan["cutoff"])
                route_plan=[(light,plan["light"]),(heavy,plan["heavy"])]
        hidden = torch.cat([stage(layer, "w13", x, ids, route_plan) for layer in shard], dim=-1)
        down = torch.cat([stage(layer, "w2", hidden, ids, route_plan) for layer in shard], dim=-1).view(m, 8, h)
        # Retain the parent grouping and the rounding of both partial sums. A real rank
        # transposes feature halves over the link before the native EP sum (see transport.py).
        pieces = []
        for rank in range(2):
            owned = (ids >= rank * group_size) & (ids < (rank + 1) * group_size)
            pieces.append(finish(down, scores, norm_weight=norm, eps=1e-8, weight_first=True,
                                 slot_expert_ids=torch.where(owned, ids, -1), expert_output_scale=shard[0].w2_post_scale))
        return pieces[0]+pieces[1], down

    fill(0)
    operations = {}
    for name, func in (("parent", run_parent), ("candidate", run_candidate)):
        func()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = func()
        operations[name] = graph, result

    def replay(name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        operations[name][0].replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    arm_order = ("parent", "candidate", "control")
    ms = {arm:[] for arm in arm_order}
    parity = []
    previous = None
    for i in range(rounds):
        fill(i + 1)
        for arm in (arm_order if i % 2 == 0 else arm_order[::-1]):
            ms[arm].append(replay("parent" if arm == "control" else arm))
        ref, bases = operations["parent"][1]
        other, down = operations["candidate"][1]
        raw_errors = 0
        for baseline_down, local in bases:
            active = local >= 0
            raw_errors += int((baseline_down[active].view(torch.int16) != down[active].view(torch.int16)).sum())
        diff = ref.float() - other.float()
        parity.append(dict(raw_mismatches=raw_errors, live_values=int((ids>=0).sum())*h,
            finite=bool(torch.isfinite(other).all() and torch.isfinite(ref).all()),
            final_bit_mismatches=int((ref.view(torch.int16)!=other.view(torch.int16)).sum()),
            changing=previous is None or not torch.equal(previous, ref),
            rel_l2=float(diff.norm()/ref.float().norm().clamp_min(1e-12))))
        previous = ref.clone()
    return dict(ms=ms, parity=parity, median_serial_ratio=statistics.median(a/b for a,b in zip(ms["parent"],ms["candidate"])))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--preserve-tiles", action="store_true")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--width", type=int, default=4096)
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--rows", type=int, nargs="+", default=[128,768])
    p.add_argument("--patterns", nargs="+", default=["uniform","skewed"])
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert hashlib.sha256((fixture.RELEASE/"candidate.diff").read_bytes()).hexdigest() == BASE
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    weights = make_weights(a.experts, a.width)
    with cached_threshold_for_preparation(HERE):
        parent = [pack(weights, i, False) for i in range(2)]
        shard = [pack(weights, i, True) for i in range(2)]
    del weights
    report = dict(baseline=BASE, width=a.width, experts=a.experts, seed=a.seed, preserve_tiles=a.preserve_tiles, complete=False, rows={},
        scope="Two original EP slices and both candidate feature slices computed serially on one SM86. Candidate gathers and norms use local memory, no H200 link. Scores apply solely to this local prototype.")
    for m in a.rows:
        for pattern in a.patterns:
            key = f"{m}_{pattern}"
            value = graphs(parent, shard, m, a.width, a.experts, a.seed, pattern, a.rounds, a.preserve_tiles)
            report["rows"][key] = value
            print(key, value["median_serial_ratio"], value["parity"][0], flush=True)
            a.out.write_text(json.dumps(report, indent=2))
    report["complete"] = True
    a.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        main()