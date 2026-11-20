"""Full 16,384-row eager prefill capture on one 6 GiB card, including peak memory."""
import argparse
import gc
import json
import types

import torch

from bench_aux import ResidualLayer, call, forward, pack_function
from build_aux import DIGEST, HERE, MODEL, PARENT, change
import hashlib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    assert hashlib.sha256((PARENT/"candidate.diff").read_bytes()).hexdigest() == DIGEST
    source = (PARENT/"candidate_sources"/MODEL).read_text()
    baseline, candidate, pack = forward(source), forward(change(source)), pack_function()
    rows, width = 16384, 8192
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    bank = torch.empty((rows+4,width), device="cuda", dtype=torch.float32).normal_(generator=gen)
    states = [bank[i:i+rows] for i in range(5)]
    embedding = states[0].to(torch.bfloat16)
    model = types.SimpleNamespace(embed_norm=None, residual_dtype=torch.float32, operator_dtype=torch.bfloat16,
        layers_to_capture=list(range(5)), layers=[ResidualLayer(t) for t in states], norm=lambda x:x)
    batch = types.SimpleNamespace(forward_mode=None)
    arms = {"baseline":baseline,"candidate":candidate,"control":baseline}
    peak = {}
    for label, fn in arms.items():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_bytes = torch.cuda.memory_allocated()
        hidden, packed = call(fn,model,batch,embedding,pack)
        torch.cuda.synchronize()
        peak[label] = torch.cuda.max_memory_allocated() - start_bytes
        for i, state in enumerate(states):
            for start in range(0, rows, 256):
                end = start + 256
                assert torch.equal(packed[start:end,i*width:(i+1)*width],state[start:end].bfloat16())
        assert torch.equal(hidden, states[-1].bfloat16())
        del packed, hidden
    times = {label:[] for label in arms}
    order = []
    for i in range(12):
        ab = ["baseline","candidate"] if i%2 == 0 else ["candidate","baseline"]
        ab.insert((0,2,2,0)[i%4],"control")
        order.append(ab)
        for key in ab:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            _ = call(arms[key], model, batch, embedding, pack)
            end.record()
            end.synchronize()
            times[key].append(1000*begin.elapsed_time(end))
            del _
    result = dict(agent="root",candidate="packed_target_taps",scope="Single eager first-turn chunk, target capture and pack",
        seed=args.seed,correct=True,active=True,complete=True,baseline_sha256=DIGEST,
        candidate_sha256=hashlib.sha256((HERE/"aux_candidate.diff").read_bytes()).hexdigest(),unit="us",
        peak_torch_allocated_delta_bytes=peak,
        fixture="five distinct contiguous overlapping views of a 512 MiB FP32 bank; every output value checked",
        bench={"instances":1,"shapes":{"M16384":{"weight":1,"times":times,"order":order}}})
    (HERE/args.out).write_text(json.dumps(result,indent=2)+"\n")
    print(peak)


if __name__ == "__main__":
    main()
