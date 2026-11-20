"""Pair registered W13 and W2 kernels using the scored graph's original route metadata."""
import torch
from humming.layer import HummingMethod


def check(rows, layer, captures, seed, pairs, candidate):
    result = {}
    for stage, calls in captures.items():
        for partition, kw in enumerate(calls):
            padded = int(kw["num_tokens_padded"][0])
            sorted_indices = kw["sorted_ids"][:padded].long()
            slots = sorted_indices[(sorted_indices >= 0) & (sorted_indices < rows * 8)]
            if not len(slots):
                continue
            kw = dict(kw)
            x = kw["inputs"] = torch.empty_like(kw["inputs"])
            out = [torch.empty_like(kw["outputs"]) for _ in range(3)]
            graphs = []
            torch.manual_seed(seed)
            x.normal_(std=.023)
            for fn,y in zip((HummingMethod.forward_layer,candidate,HummingMethod.forward_layer),out):
                kw["outputs"] = y
                fn(layer, **kw)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn(layer, **kw)
                graphs.append(g)
            times = {k:[] for k in ("baseline","candidate","control")}
            passes = []
            order = []
            for i in range(pairs):
                torch.manual_seed(seed + 211 + 37*i)
                x.normal_(std=.023)
                labels = [(0,"baseline"),(1,"candidate"),(2,"control")]
                if i%2: labels.reverse()
                order.append([name for _,name in labels])
                for index,name in labels:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[index].replay()
                    end.record()
                    end.synchronize()
                    times[name].append(start.elapsed_time(end)*1000)
                ys = [y[slots] for y in out]
                ok = torch.equal(ys[0],ys[1]) and torch.equal(ys[0],ys[2]) and bool(torch.isfinite(ys[0]).all())
                assert ok, (rows,stage,partition)
                passes.append(ok)
            result[f"{stage}_partition{partition}"]=dict(times=times, parity=passes, order=order,
                                     routed_slots=int(len(slots)),block_shape=kw["tuning_config"]["block_shape"])
            del graphs, g, x, out, ys
    assert all(any(k.startswith(s+"_") for k in result) for s in ("w13","w2")), (rows,list(result))
    return result

