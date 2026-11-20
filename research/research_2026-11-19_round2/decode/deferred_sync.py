"""Disabled research prototype: omit only a proven redundant greedy broadcast.

This does not change argmax or disable general speculative synchronization.
Startup agrees on the exact native producer and TP group. The worker creates a
single-use token immediately after that producer and consumes it before the
unchanged acceptance kernel. Dynamic sampling paths never receive a token.

Lifecycle contract: model/group implementation is immutable after startup;
changing it is a fatal configuration error, never a rank-local fallback. The
target epilogue runs on one CUDA stream, outside CUDA graph capture; replaying
the target graph still invokes a fresh native epilogue each step. Real NCCL
multi-stream and complete-server validation remain required before promotion.
"""
import os
import torch
import torch.distributed as dist


def _cuda_descriptor():
    if torch.version.hip or not torch.cuda.is_available():
        return None
    return tuple(torch.cuda.get_device_capability())


def _stream_state():
    return (torch.cuda.current_stream().cuda_stream,
            torch.cuda.is_current_stream_capturing())


def _method_is(obj, name, function):
    method = getattr(obj, name, None)
    return (getattr(method, '__self__', None) is obj and
            getattr(method, '__func__', None) is function)


class _Proof:
    __slots__ = ('owner', 'value', 'stream', 'sequence')

    def __init__(self, owner, value, stream, sequence):
        self.owner, self.value, self.stream, self.sequence = owner, value, stream, sequence


class DeferredArgmaxAgreement:
    def __init__(self, processor, sync, group, native_class, group_class, get_group):
        self.processor, self.sync, self.group = processor, sync, group
        self.native_class, self.group_class = native_class, group_class
        self.native_function = native_class.deferred_argmax
        self.sync_class, self.sync_function = type(sync), type(sync).sync
        self.sync_sites = tuple(sorted(sync._sites))
        self.get_group = get_group
        self.group_methods = tuple((name, getattr(group_class, name)) for name in
            ('all_gather_into_tensor', '_all_gather_into_tensor', 'broadcast'))
        self.ranks = tuple(group.ranks)
        self.rank = group.rank_in_group
        self.device_group, self.cpu_group = group.device_group, group.cpu_group
        self.sequence = 0
        self.pending = None

    def _check(self):
        if (type(self.processor) is not self.native_class or
                not _method_is(self.processor, 'deferred_argmax', self.native_function) or
                type(self.group) is not self.group_class or
                type(self.sync) is not self.sync_class or
                not _method_is(self.sync, 'sync', self.sync_function) or
                tuple(sorted(self.sync._sites)) != self.sync_sites or
                self.get_group() is not self.group or self.sync._tp_group is not self.group or
                tuple(self.group.ranks) != self.ranks or self.group.rank_in_group != self.rank or
                self.group.world_size != len(self.ranks) or
                self.group.device_group is not self.device_group or
                self.group.cpu_group is not self.cpu_group or
                getattr(self.group, 'pymscclpp_comm', None) is not None or
                any(not _method_is(self.group, name, function) for name, function in self.group_methods)):
            raise RuntimeError('DFLASH deferred agreement immutable producer/group contract changed')

    def produce(self, processor, sync, logits):
        if processor is not self.processor or sync is not self.sync:
            raise RuntimeError('DFLASH deferred producer or sync owner replaced')
        self._check()
        stream, capturing = _stream_state()
        # Captured outer epilogues retain the native sync; the normal target
        # graph replay has ended before this eager epilogue begins.
        result = self.processor.deferred_argmax(logits)
        self.sequence += 1
        self.pending = None
        if capturing:
            return result, None
        proof = _Proof(self, result, stream, self.sequence)
        self.pending = proof
        return result, proof

    def consume(self, proof, value, sync):
        if sync is not self.sync:
            raise RuntimeError('DFLASH deferred sync owner replaced')
        self._check()
        stream, capturing = _stream_state()
        if (type(proof) is not _Proof or proof is not self.pending or
                proof.owner is not self or proof.sequence != self.sequence or
                proof.value is not value or proof.stream != stream or capturing):
            raise RuntimeError('DFLASH stale, changed-stream, or mismatched argmax provenance')
        self.pending = None
        proof.value = None
        return True


def prepare_deferred_agreement(processor, sync):
    """Coherent startup choice; no GPU allocations or per-step CPU collectives."""
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative.spec_tp_sync import SpecTpSync
    group = sync._tp_group
    enabled = os.environ.get('SGLANG_EXPERIMENTAL_DFLASH_DEFERRED_SYNC', '0') == '1'
    cuda = _cuda_descriptor()
    native = (type(processor) is LogitsProcessor and
              _method_is(processor, 'deferred_argmax', LogitsProcessor.deferred_argmax) and
              type(group) is GroupCoordinator and get_parallel().tp_group is group and
              type(sync) is SpecTpSync and _method_is(sync, 'sync', SpecTpSync.sync) and
              all(_method_is(group, name, getattr(GroupCoordinator, name)) for name in
                  ('all_gather_into_tensor', '_all_gather_into_tensor', 'broadcast')) and
              getattr(group, 'pymscclpp_comm', None) is None)
    descriptor = (enabled, native, cuda, tuple(group.ranks), group.world_size,
                  tuple(sorted(sync._sites)))
    peers = [None] * group.world_size
    if group.world_size > 1:
        dist.all_gather_object(peers, descriptor, group=group.cpu_group)
    else:
        peers[0] = descriptor
    if (not enabled or not native or cuda is None or group.world_size <= 1 or
            any(peer != descriptor for peer in peers)):
        return None
    return DeferredArgmaxAgreement(processor, sync, group, LogitsProcessor,
                                   GroupCoordinator, lambda: get_parallel().tp_group)
