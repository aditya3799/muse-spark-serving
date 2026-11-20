"""Connect pinned SGLang packed IDs and KV translation to a small causal target.

Scheduler allocation and target weights are fixtures. All real-row target predictions
are computed with attention. Each emitted token is independently compared with a
serial GPU decoder that only sees the request's own confirmed history.
"""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import triton
import triton.language as tl
from ragged_accept import GreedyCommit

HERE = Path(__file__).resolve().parent
PIN = HERE.parent/"sglang/python/sglang"


def pinned_window():
    file = PIN/"kernels/ops/speculative/dspark/dspark_verify_window.py"
    tree = ast.parse(file.read_text())
    names = {"_compact_row_index_kernel","_compact_verify_ids_gather_kernel",
             "compact_row_index_triton","compact_verify_ids_triton"}
    nodes = [n for n in tree.body if getattr(n,"name",None) in names]
    assert len(nodes)==len(names)
    scope = dict(torch=torch,triton=triton,tl=tl,_SEARCH_NBITS=11,__name__=__name__)
    body = ast.parse("from __future__ import annotations").body + nodes
    exec(compile(ast.Module(body=body,type_ignores=[]),str(file),"exec"),scope)
    return scope


def main():
    window = pinned_window()
    module_spec = importlib.util.spec_from_file_location("trial_plan",PIN/"srt/mem_cache/kv_loc_plan.py")
    planner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(planner)
    torch.manual_seed(722)
    dev = "cuda"
    vocab, channels, batch, stride = 101, 32, 4, 32
    dimension = (vocab,channels)
    emb, values, head = [torch.randn(dimension,device=dev) for _ in range(3)]
    def next_token(seq):
        # Last-query dense causal attention over the complete serial prefix.
        e = emb[seq]
        a = (e[-1] @ e.T / channels**.5).softmax(dim=-1)
        return int(((a @ values[seq]+e[-1]*.6) @ head.T).argmax())
    histories = [torch.randint(vocab,(10+3*r,),device=dev).tolist() for r in range(batch)]
    ledger = [list(seq) for seq in histories]
    out = GreedyCommit.allocate(batch,stride,dev)
    checked = 0
    layout_kinds = [planner.IdSpaceKind.FULL, planner.IdSpaceKind.SLIDING_WINDOW]
    spaces = {kind:planner.IdSpace(key=kind,write=lambda ids,shift=i:ids*3+shift)
              for i,kind in enumerate(layout_kinds)}
    translator = SimpleNamespace(space=spaces.get)
    for round_id in range(9):
        widths = [[32,8,16,4],[8,8,8,8],[4,8,16,32]][round_id%3]
        layout = SimpleNamespace(verify_lens=torch.tensor(widths,device=dev),graph_num_tokens=96)
        prefix = torch.tensor([len(x)-1 for x in histories],device=dev)
        starts = torch.nn.functional.pad(layout.verify_lens.cumsum(0),(1,0)).int()
        ids = torch.zeros((batch,stride),device=dev,dtype=torch.int64)
        for r,(hist,width) in enumerate(zip(histories,widths)):
            seq = hist.copy()
            candidates = [seq[-1]]
            # Force a chosen mix of rejection positions with the serial oracle.
            cutoff = (round_id*5+r*3) % width
            for j in range(width-1):
                token = next_token(torch.tensor(seq,device=dev))
                if j == cutoff:
                    token = (token+1)%vocab
                candidates.append(token)
                seq.append(token)
            ids[r,:width] = torch.tensor(candidates,device=dev)
        draft = torch.nn.functional.pad(ids[:,1:],(0,1))
        packed = window["compact_verify_ids_triton"](draft_block_ids=ids,draft_tokens=draft,
                                                       layout=layout,device=dev)
        req,col,valid = window["compact_row_index_triton"](verify_lens=layout.verify_lens,
                                                           padded_total=layout.graph_num_tokens,device=dev)
        window_cols = torch.where(valid,req*stride+col,-1)
        virtual = torch.randperm(batch*stride,device=dev)+1
        plan = planner.KVLocPlan(source=translator,req_pool_indices=torch.arange(batch,device=dev),
                                 seq_lens=prefix,seq_lens_cpu=None,write_virtual=virtual,read_extent=stride)
        full,swa = [plan.write_ids(translator,kind=k,cols=window_cols) for k in layout_kinds]
        assert torch.all(full[~valid]==0) and torch.all(swa[~valid]==0)
        # Build one packed attention matrix with each request's previous prefix
        # and its proposed chain. Every row attends only to its own causal keys.
        old_keys = torch.tensor([tok for seq in histories for tok in seq[:-1]],device=dev)
        owners = torch.arange(batch,device=dev).repeat_interleave(prefix)
        keys = torch.cat((old_keys,packed))
        key_owner = torch.cat((owners,req))
        old_pos = torch.tensor([j for seq in histories for j in range(len(seq)-1)],device=dev)
        query_pos = prefix[req.clamp(max=batch-1)]+col
        key_pos = torch.cat((old_pos,query_pos))
        mask = (req[:,None]==key_owner)&(key_pos<=query_pos[:,None])
        weights = ((emb[packed] @ emb[keys].T) / channels**.5).masked_fill(~mask,-torch.inf).softmax(1)
        predicted = ((weights @ values[keys]+emb[packed]*.6) @ head.T).argmax(1).contiguous()
        out.verify(packed,predicted,starts,prefix,full)
        for r,width in enumerate(widths):
            n = int(out.lengths[r])
            assert 1 <= n <= width
            source = out.sources[r,:n].long()
            target_window = torch.arange(r*stride,r*stride+n,device=dev)
            assert torch.equal(window_cols[source],target_window)
            assert torch.equal(out.retained_slots[r,:n],virtual[target_window]*3)
            assert torch.equal(swa[source],virtual[target_window]*3+1)
            for token in out.emitted[r,:n].tolist():
                assert next_token(torch.tensor(ledger[r],device=dev))==token
                ledger[r].append(token)
                checked += 1
            histories[r].extend(out.emitted[r,:n].tolist())
            assert ledger[r]==histories[r]
    result=dict(tokens_compared=checked,decode_iterations=9,requests=batch,
                target="seeded single-layer causal attention fixture, GPU fp32",
                pinned="compact_row_index_triton, compact_verify_ids_triton, KVLocPlan.write_ids",
                device=torch.cuda.get_device_name())
    (HERE/"packed_target_trial.json").write_text(json.dumps(result,indent=2))
    print(result)


if __name__ == "__main__":
    main()

