from pathlib import Path
H=Path(__file__).resolve().parent
def change(f,changes):
    p=H/f;s=p.read_text(encoding="utf-8-sig")
    for a,b in changes:
        if a not in s:
            assert b in s,(f,a)
            continue
        s=s.replace(a,b,1)
    p.write_text(s,encoding="utf8")
change("fixed_tree_adapter.py",[
("from tree_commit import _commit","from tree_commit import _commit\nfrom tree_compact_pages import fill_compact_swa"),
("def __init__(self, batch, capacity, device):","def __init__(self, batch, capacity, device, window=None):"),
("        self.swa = torch.empty_like(self.full)","""        self.window = window
        self.swa = alloc((batch*WIDTH,capacity if window is None else min(capacity,window+1)),torch.int32)
        self.swa_lengths = torch.empty_like(self.lengths)
        self.swa_cu_k = torch.zeros_like(self.cu_k)"""),
('            if key == "batch":','            if key == "window":\n                view.window = value\n                continue\n            if key == "batch":'),
('if key in ("cu_q","cu_k"):', 'if key in ("cu_q","cu_k","swa_cu_k"):'),
('"positions","full","swa")','"positions","full","swa","swa_lengths")'),
("""        fill_tree_suffix(self.swa,self.prefix_pages,self.ancestors,prefix_lens,
            self.ancestor_lens,swa_v2p,window,WIDTH)""","""        if self.window is None:
            fill_tree_suffix(self.swa,self.prefix_pages,self.ancestors,prefix_lens,
                self.ancestor_lens,swa_v2p,window,WIDTH)
        else:
            if window!=self.window:
                raise ValueError("Window changed after the slab was allocated")
            fill_compact_swa(self.swa,self.prefix_pages,self.ancestors,prefix_lens,
                self.ancestor_lens,swa_v2p,self.swa_lengths,window,WIDTH)
            torch.cumsum(self.swa_lengths,0,dtype=torch.int32,out=self.swa_cu_k[1:])"""),
("        return flash_attn_with_kvcache(q=",'''        compact = sliding and self.window is not None
        if compact and layer.sliding_window_size != self.window:
            raise ValueError("Per-layer window differs from the guarded Muse model window")
        return flash_attn_with_kvcache(q='''),
("cache_seqlens=self.lengths,cu_seqlens_q=self.cu_q,cu_seqlens_k_new=self.cu_k,","cache_seqlens=self.swa_lengths if compact else self.lengths,cu_seqlens_q=self.cu_q,\n            cu_seqlens_k_new=self.swa_cu_k if compact else self.cu_k,")])
change("build_fixed_tree.py",[
('put(path,"import triton','code+="\\n"+(H/"tree_compact_pages.py").read_text(encoding="utf-8-sig")+"\\n"\nput(path,"import triton'),
('put(S+"fixed_tree_adapter.py",adapter)','adapter=adapter.replace("from tree_compact_pages import fill_compact_swa","from sglang.srt.speculative.fixed_tree_kernels import fill_compact_swa")\nput(S+"fixed_tree_adapter.py",adapter)')])
change("tree_dispatch.py",[
("from sglang.srt.runtime_context import get_parallel","from sglang.srt.runtime_context import get_parallel\nfrom sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite"),
("slab=TreeStep(bs_cap,cap,table.device)","slab=TreeStep(bs_cap,cap,table.device,backend.sliding_window_size)"),
("    accept,step.counts","    prediction=prediction.contiguous().view(step.batch,WIDTH)\n    worker._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY,prediction)\n    accept,step.counts")])
change("test_fixed_tree_adapter.py",[('step=TreeStep(b,table.shape[1],"cpu")','step=TreeStep(b+1,32,"cpu",window=4).active(b)')])

