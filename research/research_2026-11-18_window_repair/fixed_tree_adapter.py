"""Fixed-eight page-one tree adapter. Supply the four frozen planner kernels on PYTHONPATH."""
import torch
from fused_tree import _grow
from tree_expand import _expand
from tree_pages import fill_tree_suffix
from tree_commit import _commit
from tree_compact_pages import fill_compact_swa

WIDTH = 8


class TreeStep:
    """One exclusive in-flight workspace, including outputs until scheduler consumption."""

    def __init__(self, batch, capacity, device, window=None):
        self.batch = batch
        def alloc(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)
        shape = (batch, WIDTH)
        self.bind_topology(alloc((batch*WIDTH*5,),torch.int32))
        self.ancestors = alloc((batch * WIDTH, WIDTH), torch.int32)
        self.ancestor_lens = alloc((batch * WIDTH,), torch.int32)
        self.lengths = torch.empty_like(self.ancestor_lens)
        self.positions = alloc((batch * WIDTH,), torch.int64)
        self.cu_q = torch.arange(batch * WIDTH + 1, dtype=torch.int32, device=device)
        self.cu_k = torch.zeros_like(self.cu_q)
        self.prefix_pages = alloc((batch, capacity), torch.int32)
        self.full = alloc((batch * WIDTH, capacity), torch.int32)
        self.window = window
        self.swa = alloc((batch*WIDTH,capacity if window is None else min(capacity,window+1)),torch.int32)
        self.swa_lengths = torch.empty_like(self.lengths)
        self.swa_cu_k = torch.zeros_like(self.cu_k)
        self.all_rows = torch.full((batch,), WIDTH, device=device, dtype=torch.int32)
        self.emitted = alloc(shape, torch.int64)
        self.counts = torch.empty_like(self.all_rows)
        self.bonus = alloc((batch,), torch.int64)
        self.src_at_dest = alloc(shape, torch.int32)
        self.new_lens = torch.empty_like(self.all_rows)

    def bind_topology(self, wire):
        self.wire=wire
        n=self.batch*WIDTH
        self.tokens=wire[:2*n].view(torch.int64).view(self.batch,WIDTH)
        self.parents=wire[2*n:3*n].view(self.batch,WIDTH)
        self.depth=wire[3*n:4*n].view(self.batch,WIDTH)
        self.mask=wire[4*n:5*n].view(torch.uint32).view(self.batch,WIDTH)

    def active(self, batch):
        """View a single allocator-owned slab across all eager batch sizes."""
        if not 0 < batch <= self.batch:
            raise ValueError("Batch exceeds the slab capacity")
        view = object.__new__(type(self))
        view.batch = batch
        view.bind_topology(self.wire[:batch*WIDTH*5])
        for key,value in vars(self).items():
            if key in ("wire","tokens","parents","depth","mask"):
                continue
            if key == "window":
                view.window = value
                continue
            if key == "batch":
                continue
            if key in ("cu_q","cu_k","swa_cu_k"):
                rows = batch * WIDTH + 1
            elif key in ("ancestors","ancestor_lens","lengths","positions","full","swa","swa_lengths"):
                rows = batch * WIDTH
            else:
                rows = batch
            setattr(view,key,value[:rows])
        return view

    def propose(self, ids, conditional_logits, anchors, inv_temp=1., log_fork=0.):
        if ids.shape != (self.batch, 7, 16) or conditional_logits.shape != (*ids.shape,16):
            raise ValueError("Expected the native seven-position, top-16 selector lattice")
        if not all(x.is_contiguous() for x in (ids, conditional_logits, anchors)):
            raise ValueError("Selector inputs must be contiguous")
        _grow[(self.batch,)](ids, conditional_logits, anchors, anchors, self.tokens,
            self.parents,self.depth,self.mask,inv_temp,log_fork,7,16,WIDTH,WIDTH,False,num_warps=4)
        return self.tokens

    def stage(self, table, req_ids, prefix_lens, owned_virtual, full_v2p, swa_v2p, window):
        """All input ids are virtual, including the kernel's copied reservation.
        Both maps must correspond to this batch's translator generation.
        Supply a prebuilt identity map for an identity pool. No allocation in the cache.
        """
        b = self.batch
        if (table.dtype != self.prefix_pages.dtype or table.shape[1] < self.prefix_pages.shape[1]
            or owned_virtual.shape != (b, WIDTH) or not owned_virtual.is_contiguous()
            or not table.is_contiguous() or window < 1):
            raise ValueError("Incompatible request table, page-one reservation or sliding window")
        torch.index_select(table[:, :self.prefix_pages.shape[1]], 0, req_ids, out=self.prefix_pages)
        _expand[(b*WIDTH,)](owned_virtual,self.mask,self.depth,prefix_lens,
            self.ancestors,self.ancestor_lens,self.lengths,self.positions,WIDTH)
        torch.cumsum(self.lengths, 0, dtype=torch.int32, out=self.cu_k[1:])
        fill_tree_suffix(self.full,self.prefix_pages,self.ancestors,prefix_lens,
            self.ancestor_lens,full_v2p,self.full.shape[1],WIDTH)
        if self.window is None:
            fill_tree_suffix(self.swa,self.prefix_pages,self.ancestors,prefix_lens,
                self.ancestor_lens,swa_v2p,window,WIDTH)
        else:
            if window!=self.window:
                raise ValueError("Window changed after the slab was allocated")
            fill_compact_swa(self.swa,self.prefix_pages,self.ancestors,prefix_lens,
                self.ancestor_lens,swa_v2p,self.swa_lengths,window,WIDTH)
            torch.cumsum(self.swa_lengths,0,dtype=torch.int32,out=self.swa_cu_k[1:])

    def attention(self, backend, layer, q, flash_attn_with_kvcache, out=None):
        """Post-KV-write native FA3 MHA call for a dedicated page-one TARGET_VERIFY path.
        Only standard causal Muse layers without custom score modifiers, sinks or CP.
        This method needs its own graph variant or an eager-only target dispatch.
        """
        if (backend.page_size != 1 or backend.fa_impl_ver != 3 or backend.use_mla
            or backend.fa_skip_kv_cache or backend.kv_cache_is_mxfp8 or layer.is_cross_attention):
            raise ValueError("Unsupported attention configuration")
        q, _, _, kd, vd = backend.prepare_paged_mha_query(q,None,None,layer,
            logical_batch_size=self.batch*WIDTH,kv_head_num=layer.tp_k_head_num,is_prefill=True)
        kc, vc = backend.get_paged_mha_kv_cache(layer)
        scales = {} if kd is None else dict(k_descale=kd,v_descale=vd)
        sliding = layer.sliding_window_size is not None and layer.sliding_window_size >= 0
        compact = sliding and self.window is not None
        if compact and layer.sliding_window_size != self.window:
            raise ValueError("Per-layer window differs from the guarded Muse model window")
        return flash_attn_with_kvcache(q=q.contiguous().view(-1,layer.tp_q_head_num,layer.head_dim),
            k_cache=kc,v_cache=vc,page_table=self.swa if sliding else self.full,
            cache_seqlens=self.swa_lengths if compact else self.lengths,cu_seqlens_q=self.cu_q,
            cu_seqlens_k_new=self.swa_cu_k if compact else self.cu_k,
            max_seqlen_q=1,softmax_scale=layer.scaling,causal=True,
            window_size=(layer.sliding_window_size,0) if sliding else (-1,-1),
            softcap=layer.logit_cap,return_softmax_lse=False,num_splits=backend.num_splits,
            out=out,ver=3,**scales)

    def commit(self, target_top1, table, req_ids, prefix_lens, owned_virtual):
        """Use the immutable original reservation, independent of tables overwritten by this launch."""
        if target_top1.numel() != self.batch*WIDTH or not target_top1.is_contiguous():
            raise ValueError("Need one contiguous target argmax per node")
        _commit[(self.batch,)](self.tokens,self.parents,target_top1,owned_virtual,req_ids,
            prefix_lens,self.all_rows,table,self.emitted,self.counts,self.bonus,self.src_at_dest,
            self.new_lens,table.stride(0),self.batch,WIDTH)

    def append_and_publish(self, worker, plan, target_hidden, publish, war_event):
        """Call once after commit on the same forward stream. All reserved rows are private.
        Explicitly guard correction/mamba paths upstream, before drafting/target execution.
        """
        if worker.lilicorr is not None or worker._need_mamba_verify_commit:
            raise ValueError("The ordinal correction and Mamba state require a path-specific commit")
        cache_loc = worker._draft_write_ids(plan)
        worker._append_target_hidden_to_draft_kv_by_loc(target_hidden=target_hidden,
            cache_loc=cache_loc,cache_loc_2d=cache_loc.view(self.batch,WIDTH),
            positions=self.positions,commit_lens=self.all_rows)
        # The scheduler ordinarily consumes a read event from the target graph. Its early signal
        # does not cover the request-table write above. Replace it before the scheduler gets control.
        war_event.record()
        worker.model_runner.shared_read_done_event = war_event
        if publish is not None:
            publish(self.new_lens)

