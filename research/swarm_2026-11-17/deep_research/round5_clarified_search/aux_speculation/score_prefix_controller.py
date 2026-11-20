"""Default-off research score export. No SGLang imports, server hooks or CUDA policy.

A margin is a model score statistic, NOT a calibrated acceptance probability.
The exporter gathers rows along the actual proposed path, not independent slots.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import threading
import torch


def binding(t):
    return (id(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype,
            str(t.device), t.storage_offset(), t.untyped_storage().nbytes())


def interval(t):
    return (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())


def export_path_margins(candidate_ids, scores, tokens, out):
    """GPU-compatible tensor operations; only static shape/device checks on host.

    out[B,8]: seven selected-minus-best-other margins, then validity as FP32.
    Input scores[B,7,K,K], candidate_ids[B,7,K], tokens[B,7].
    Invalid/nonfinite/non-greedy/ambiguous path sets validity0; policy falls back8.
    No score softmax and no approximation to the target distribution is used.
    """
    b, gamma, k = candidate_ids.shape
    if gamma != 7 or k < 2 or tuple(scores.shape) != (b,7,k,k):
        raise ValueError('fixed draft8 with at least two candidates required')
    if tuple(tokens.shape) != (b,7) or tuple(out.shape) != (b,8):
        raise ValueError('token/output shape mismatch')
    if candidate_ids.dtype != torch.int64 or tokens.dtype != torch.int64:
        raise TypeError('token ids must be int64')
    if scores.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError('score boundary must be FP32')
    if any(t.device != out.device or not t.is_contiguous()
           for t in (candidate_ids,scores,tokens,out)):
        raise ValueError('same device and contiguous inputs/output required')
    lo, hi = interval(out)
    if any(max(lo,interval(t)[0]) < min(hi,interval(t)[1])
           for t in (candidate_ids,scores,tokens)):
        raise ValueError('output aliases input')
    match = candidate_ids == tokens.unsqueeze(-1)
    path = match.to(torch.int64).argmax(-1)
    predecessor = torch.cat((torch.zeros_like(path[:,:1]),path[:,:-1]),1)
    rows = scores.gather(2, predecessor[:,:,None,None].expand(b,7,1,k))[:,:,0,:]
    chosen = rows.gather(-1,path.unsqueeze(-1)).squeeze(-1)
    others = rows.masked_fill(match, float('-inf')).amax(-1)
    margins = chosen - others
    valid = ((match.sum(-1)==1).all(-1) & torch.isfinite(rows).all(-1).all(-1)
             & (rows.argmax(-1)==path).all(-1) & torch.isfinite(margins).all(-1))
    out[:,:7].copy_(margins)
    out[:,7].copy_(valid.to(torch.float32))


@dataclass(frozen=True)
class Snapshot:
    epoch: int
    generation: int
    request_key: str
    margins: tuple[float,...]
    valid: bool
    proposed_tokens: tuple[int,...] = ()


class ScoreExport:
    """Single owner, single serving stream; host consumes before next replay.

    The graph owns strong buffer refs through this object; keep it alive until
    graph retirement is synchronized. Invalidate before weight update/recapture.
    record/lease APIs are host orchestration, not code to execute inside replay.
    CPU tests do not establish CUDA event/stream ordering correctness.
    """
    def __init__(self, *, device='cpu', epoch=0):
        self.out=torch.empty((1,8),device=device,dtype=torch.float32)
        self.tokens=torch.empty((1,7),device=device,dtype=torch.int64)
        self.fixed_binding=(binding(self.out),binding(self.tokens))
        self.epoch=epoch; self.generation=0; self.phase='new'
        self.graph=None; self.stream=None; self.request_key=None
        self.thread=threading.get_ident(); self.invalid_reason=None
        self.capture_writes=0
        self.pending_graph=None; self.capture_succeeded=False

    def guard(self):
        if self.invalid_reason is not None:
            raise RuntimeError('invalidated: '+self.invalid_reason)
        if threading.get_ident()!=self.thread or (binding(self.out),binding(self.tokens))!=self.fixed_binding:
            raise RuntimeError('owner thread or immutable tensor binding changed')

    def begin_capture(self):
        self.guard()
        if self.phase!='new': raise RuntimeError('fresh owner required for capture')
        self.phase='capturing'

    def capture_write(self,candidate_ids,scores,tokens):
        self.guard()
        if self.phase not in ('capturing','inflight'):
            raise RuntimeError('capture/write outside producer lease')
        if self.out.is_cuda and self.phase=='capturing':
            if self.pending_graph is None or not torch.cuda.is_current_stream_capturing():
                raise RuntimeError('export is not inside managed CUDA capture')
        export_path_margins(candidate_ids,scores,tokens,self.out)
        self.tokens.copy_(tokens)
        if self.phase=='capturing': self.capture_writes+=1

    def finish_capture(self,graph,stream):
        self.guard()
        if self.phase!='capturing' or graph is None or self.capture_writes<1:
            raise RuntimeError('bad capture or missing export write')
        if self.out.is_cuda and (graph is not self.pending_graph or not self.capture_succeeded):
            raise RuntimeError('successful matching managed CUDA graph required')
        self.graph=graph; self.stream=stream; self.phase='idle'

    def capture_cuda(self,producer,*,capture_stream=None,pool=None):
        """Own graph construction/capture; warmup must run before attachment.

        On failure retain graph/buffers and invalidate; caller synchronizes before
        retirement. This method must run on the intended serving stream.
        """
        if not self.out.is_cuda: raise RuntimeError('CUDA storage required')
        self.guard()
        serving_stream=torch.cuda.current_stream(self.out.device).cuda_stream
        self.begin_capture()
        graph=torch.cuda.CUDAGraph();self.pending_graph=graph
        try:
            with torch.cuda.graph(graph,stream=capture_stream,pool=pool):
                producer()
            self.capture_succeeded=True
            self.finish_capture(graph,serving_stream)
        except BaseException:
            self.abort('CUDA capture failed; synchronize before retirement')
            raise
        return graph

    def replay_cuda_blocking(self,request_key):
        """Known captured graph only, explicit score AND token D2H transfers."""
        if not self.out.is_cuda: raise RuntimeError('CUDA storage required')
        stream=torch.cuda.current_stream(self.out.device).cuda_stream
        generation=self.begin_replay(self.graph,stream,request_key)
        try:
            self.graph.replay()
            self.finish_replay_enqueue(generation)
            return self.consume_blocking(generation,stream,request_key)
        except BaseException:
            self.abort('replay/readback failed; synchronize before retirement')
            raise

    def actual_stream_guard(self,stream):
        if self.out.is_cuda and torch.cuda.current_stream(self.out.device).cuda_stream!=stream:
            raise RuntimeError('actual CUDA stream differs from bound serving stream')

    def begin_replay(self,graph,stream,request_key):
        self.guard()
        if self.phase!='idle' or graph is not self.graph or stream!=self.stream:
            raise RuntimeError('pending lease, stale graph or wrong stream')
        if not isinstance(request_key,str) or not request_key:
            raise ValueError('explicit current-request identity required')
        self.actual_stream_guard(stream)
        self.generation+=1; self.request_key=request_key; self.phase='inflight'
        return self.generation

    def finish_replay_enqueue(self,generation):
        self.guard()
        if self.phase!='inflight' or generation!=self.generation:
            raise RuntimeError('stale producer generation')
        self.actual_stream_guard(self.stream)
        self.phase='ready'

    def consume_blocking(self,generation,stream,request_key):
        """Explicit D2H synchronization cost. No async readiness is implied.

        Caller must invoke on the same actual CUDA stream as enqueue. The key
        argument is a CPU fixture contract, not a replacement for CUDA stream APIs.
        """
        self.guard()
        if (self.phase!='ready' or generation!=self.generation or stream!=self.stream
                or request_key!=self.request_key):
            raise RuntimeError('stale request/generation or wrong stream')
        self.actual_stream_guard(stream)
        # Clone independent storage, including CPU: later replay cannot mutate it.
        host=self.out.detach().to('cpu',copy=True)
        values=host[0].tolist()
        # Separate exact int64 token readback: count BOTH copies in timing.
        host_tokens=tuple(self.tokens.detach().to('cpu',copy=True)[0].tolist())
        self.phase='idle'
        return Snapshot(self.epoch,generation,request_key,tuple(values[:7]),values[7]==1.,host_tokens)

    def abort(self,reason):
        # A queued/failed CUDA producer might still write. Never recycle this owner.
        self.invalid_reason=str(reason); self.phase='invalid'


@dataclass(frozen=True)
class Policy:
    enabled: bool=False
    tail_margin_threshold: float|None=None
    policy_version: str='uncalibrated-research-v1'

    def signature(self):
        return (self.enabled,self.tail_margin_threshold,self.policy_version)

    def choose(self,snapshot):
        validate_snapshot(snapshot)
        if not self.enabled: return 8
        if (self.tail_margin_threshold is None
                or not math.isfinite(self.tail_margin_threshold)
                or self.tail_margin_threshold < 0):
            raise ValueError('explicit finite nonnegative research threshold required')
        if not snapshot.valid or any(not math.isfinite(v) for v in snapshot.margins):
            return 8
        # W4 ends before proposal edge3; all four additional edges matter to W8.
        return 4 if min(snapshot.margins[3:]) < self.tail_margin_threshold else 8


@dataclass(frozen=True)
class Packet:
    epoch: int
    generation: int
    request_key: str
    policy_signature: tuple
    proposed_tokens: tuple[int,...]
    width: int


def validate_snapshot(snapshot):
    if (type(snapshot.epoch) is not int or snapshot.epoch<0
            or type(snapshot.generation) is not int or snapshot.generation<1
            or not isinstance(snapshot.request_key,str) or not snapshot.request_key
            or type(snapshot.valid) is not bool or len(snapshot.margins)!=7):
        raise ValueError('malformed snapshot metadata')
    if any(type(v) not in (float,int) for v in snapshot.margins):
        raise ValueError('margins must be numeric')
    if snapshot.valid and any(not math.isfinite(v) for v in snapshot.margins):
        raise ValueError('valid snapshot cannot contain nonfinite margins')


def make_packet(snapshot,policy,proposed_tokens):
    validate_snapshot(snapshot)
    if len(proposed_tokens)!=7 or any(type(t) is not int or t<0 for t in proposed_tokens):
        raise ValueError('seven actual nonnegative integer proposed tokens required')
    if tuple(proposed_tokens)!=snapshot.proposed_tokens:
        raise RuntimeError('proposal differs from captured score path')
    return Packet(snapshot.epoch,snapshot.generation,snapshot.request_key,
                  policy.signature(),tuple(proposed_tokens),policy.choose(snapshot))


def agree_packets(packets,world_size):
    """CPU model of all-gather + deterministic decision, NOT distributed code.

    Every rank supplies a packet every step. Inconsistent identity/policy/proposal
    is a collective error (cannot safely run target on different requests/tokens).
    Width-only disagreement uses rank0: its own request scores are canonical.
    """
    if len(packets)!=world_size or world_size<1: raise RuntimeError('incomplete ranks')
    first=packets[0]
    signature=lambda p:(p.epoch,p.generation,p.request_key,p.policy_signature,p.proposed_tokens)
    if any(signature(p)!=signature(first) or p.width not in (4,8) for p in packets):
        raise RuntimeError('collective identity/policy/token contract mismatch')
    return first.width




