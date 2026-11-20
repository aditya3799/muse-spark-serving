"""Replay the exact target capture and packing code with fixed residual snapshots."""
import argparse
import ast
import hashlib
import json


import types
from pathlib import Path
from typing import Optional

import torch

from build_aux import HERE, ROOT, PARENT, MODEL, DIGEST, change

# Other model layers supply these snapshots in production. Their arithmetic is excluded.
class ResidualLayer:
    def __init__(self, state):
        self.state = state

    def self_attn_input_norm(self, stream):
        return stream

    def __call__(self, positions, stream, attn_input, batch):
        return self.state, attn_input


def forward(source):
    tree = ast.parse(source)
    model = next(node for node in tree.body if getattr(node, "name", None) == "MuseSpark1xModel")
    method = next(node for node in model.body if getattr(node, "name", None) == "forward")
    ctx = types.SimpleNamespace(set=lambda *args: None)
    ns = dict(torch=torch, Optional=Optional, ForwardBatch=object, get_forward=lambda: ctx,
              layernorm_sp=types.SimpleNamespace(runs_sp=lambda mode: False))
    exec(compile(ast.Module(body=[method], type_ignores=[]), MODEL, "exec"), ns)
    return ns["forward"]


def pack_function():
    import subprocess
    path = "python/sglang/srt/layers/aux_hidden_states.py"
    code = subprocess.check_output(["git", "-C", str(ROOT/"sglang"), "show", "297e54fd:" + path]).decode()
    node = next(n for n in ast.parse(code).body if getattr(n, "name", "") == "pack_aux_hidden_states")
    ns = dict(torch=torch, AuxHiddenStates=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns["pack_aux_hidden_states"]


def call(fn, model, batch, embedding, pack):
    output = fn(model, None, None, batch, input_embeds=embedding)
    if isinstance(output, tuple):
        hidden, aux = output
        return hidden, pack(aux)
    return (output,)


def case(rows, seed, device, rounds, forward_a, forward_b, pack, fc_weight=None):
    width = 8192 if device == "cuda" else 64
    gen = torch.Generator(device=device).manual_seed(seed + rows)
    snapshots = [torch.randn((rows, width), device=device, generator=gen) for _ in range(5)]
    embedding = snapshots[0].to(torch.bfloat16)
    batch = types.SimpleNamespace(forward_mode=None)
    model = types.SimpleNamespace(embed_norm=None, residual_dtype=torch.float32, operator_dtype=torch.bfloat16,
                                  layers=[ResidualLayer(x) for x in snapshots], norm=lambda x: x)
    for taps in ([], [2], [1, 1, 4], [0, 3, 4], [0, 1, 2, 3, 4]):
        model.layers_to_capture = taps
        left = call(forward_a, model, batch, embedding, pack)
        right = call(forward_b, model, batch, embedding, pack)
        assert len(left) == len(right) and all(torch.equal(a,b) for a,b in zip(left, right))
    assert isinstance(forward_b(model, None, None, batch, input_embeds=embedding)[1], torch.Tensor)
    del left, right
    if device == "cpu":
        return
    if fc_weight is not None:
        native_pack = pack
        def pack(aux):
            return torch.nn.functional.linear(native_pack(aux), fc_weight)
    arms = {"baseline": forward_a, "control": forward_a, "candidate": forward_b}
    graphs = {}
    outputs = {}
    for key, fn in arms.items():
        with torch.cuda.stream(torch.cuda.Stream()):
            call(fn, model, batch, embedding, pack)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outputs[key] = call(fn, model, batch, embedding, pack)
        graphs[key] = g
    # Mutate inputs after capture and check all final outputs.
    snapshots[2].neg_()
    oracle = call(forward_a, model, batch, embedding, pack)
    for key, g in graphs.items():
        g.replay()
        assert all(torch.equal(a,b) for a,b in zip(oracle,outputs[key]))
    for g in graphs.values():
        for _ in range(3):
            g.replay()
    torch.cuda.synchronize()
    repeats = 1 if fc_weight is not None else (12 if rows <= 256 else 3)
    times = {key: [] for key in arms}
    orders = []
    for i in range(rounds):
        ab = ["baseline", "candidate"] if i % 2 == 0 else ["candidate", "baseline"]
        ab.insert((0,2,2,0)[i % 4], "control")
        orders.append(ab)
        for key in ab:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repeats):
                graphs[key].replay()
            end.record()
            end.synchronize()
            times[key].append(1000 * begin.elapsed_time(end) / repeats)
    return dict(weight=1, times=times, order=orders, captured_values=5*rows*width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--with-fc", action="store_true")
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    assert hashlib.sha256((PARENT/"candidate.diff").read_bytes()).hexdigest() == DIGEST
    source = (PARENT / "candidate_sources" / MODEL).read_text()
    fns, pack = (forward(source), forward(change(source))), pack_function()
    if args.cpu:
        for n in (0, 1, 5):
            case(n, args.seed, "cpu", args.rounds, *fns, pack)
        print("15 native forward and packing CPU cases passed")
        return
    assert args.output and args.rounds % 4 == 0
    result = dict(agent="root", candidate="packed_target_taps", scope="Target capture and final pack, 5x8192 BF16",
                  candidate_sha256=hashlib.sha256((HERE/"aux_candidate.diff").read_bytes()).hexdigest(),
                  seed=args.seed, baseline_sha256=DIGEST, active=True, correct=True, gpu=torch.cuda.get_device_name(),
                  unit="us", bench={"instances":1,"shapes":{}}, complete=False)
    fc_weight = None
    rows_to_run = (8, 32, 128, 256, 753, 2048, 4096)
    if args.with_fc:
        result["scope"] += " plus native full-K 40960-to-8192 draft projection, excludes RMSNorm and KV writes"
        rows_to_run = (8, 32, 128, 256, 753)
        generator = torch.Generator(device="cuda").manual_seed(args.seed + 9231)
        fc_weight = torch.empty((8192,40960), device="cuda", dtype=torch.bfloat16)
        fc_weight.normal_(mean=0, std=0.005, generator=generator)
    for rows in rows_to_run:
        entry = case(rows, args.seed, "cuda", args.rounds, *fns, pack, fc_weight)
        result["bench"]["shapes"][f"M{rows}"] = entry
        print(rows, flush=True)
        torch.cuda.empty_cache()
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
