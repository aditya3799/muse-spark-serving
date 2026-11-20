"""Replay a branch commit against independent token walking and fragmented target/draft KV fixtures."""
import argparse
import json
import os
import random
from pathlib import Path
import torch
from selector_tree import walk
from tree_commit import _commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.device == "cpu":
        assert os.environ.get("TRITON_INTERPRET") == "1"
    torch.set_num_threads(1)
    rng = random.Random(847193)
    checked = 0
    tail = 0
    noncontiguous = 0
    for width in (8, 16, 32):
        stride = 270
        nreq = 9
        table = torch.randperm(nreq*stride, generator=torch.Generator().manual_seed(width)).int().view(nreq,stride)
        # Independent physical placements represent target full, target sliding, and draft KV.
        tables = [torch.randperm(nreq*stride, generator=torch.Generator().manual_seed(width+i)).long() for i in range(3)]
        dev = args.device
        table_device = table.to(dev)
        for trial in range(22):
            b = 7
            live = trial % (b+1)
            req = rng.sample(range(nreq), b)
            pref = [rng.choice((2, 17, 79, 190)) for _ in req]
            budgets = [rng.choice((0, 1, 2, width-1, width, width+17)) for _ in req]
            branches = []
            toks = []
            pred = []
            gold = []
            before = table.clone()
            slots = torch.stack([table[r,p:p+width] for r,p in zip(req,pref)])
            stores = [torch.full((nreq*stride,), -79001, dtype=torch.long) for _ in tables]
            for j in range(b):
                parents = [0] + [rng.randrange(i) for i in range(1, width)]
                tokens = rng.sample(range(1200, 9000), width)
                target = []
                for i in range(width):
                    children = [k for k in range(1,width) if parents[k] == i]
                    target.append(tokens[rng.choice(children)] if children and rng.random() < 0.86 else -10-i)
                branches.append(parents)
                toks.append(tokens)
                pred.append(target)
                gold.append(walk(tokens, parents, target, budgets[j]))
                for space,(ptr,cache) in enumerate(zip(tables,stores)):
                    cache[ptr[slots[j].long()]] = torch.arange(width)*17 + space*7 + j*1300
            def g(x, dtype):
                return torch.as_tensor(x, dtype=dtype, device=dev)
            def out(shape, dtype):
                return torch.full(shape, -199, device=dev, dtype=dtype)
            emitted = out((b,width),torch.long)
            counts = out((b,),torch.int32)
            bonus = out((b,),torch.long)
            order = out((b,width),torch.int32)
            sequence = out((b,),torch.int32)
            _commit[(b,)](g(toks,torch.long),g(branches,torch.int32),g(pred,torch.long),
                slots.to(dev),g(req,torch.int64),g(pref,torch.int32),g(budgets,torch.int32),
                table_device,emitted,counts,bonus,order,sequence,stride,live,width,num_warps=1)
            for j in range(live):
                out_tokens, keep = gold[j]
                permutation = keep + [k for k in range(width) if k not in keep]
                assert emitted[j].tolist() == out_tokens + [0]*(width-len(out_tokens))
                assert order[j].tolist() == permutation
                assert int(counts[j]) == len(keep)
                assert int(sequence[j]) == pref[j]+len(keep)
                assert int(bonus[j]) == (out_tokens[-1] if out_tokens else -1)
                selected_slots = table_device[req[j],pref[j]:pref[j]+width].cpu().long()
                assert selected_slots.tolist() == slots[j,permutation].tolist()
                for space,(ptr,cache) in enumerate(zip(tables,stores)):
                    assert cache[ptr[selected_slots[:len(keep)]]].tolist() == [k*17+space*7+j*1300 for k in keep]
                table[req[j],pref[j]:pref[j]+width] = slots[j,permutation]
                checked += 1
                tail += width-len(keep)
                noncontiguous += keep != list(range(len(keep)))
            assert torch.equal(table_device.cpu(),table)
            for output in (emitted,counts,bonus,order,sequence):
                assert bool((output[live:]==-199).all())
            assert torch.equal(table.sort().values,before.sort().values)
    result = dict(device=args.device, live_requests=checked, retained_noncontiguous=noncontiguous,
                  rejected_slots_checked=tail, verified_full_sliding_and_draft_spaces=3)
    print(result)
    Path(__file__).with_name("tree_commit_"+args.device+".json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
