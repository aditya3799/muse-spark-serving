"""Local storage/metadata gate, explicitly NOT Hopper native-FP8 qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from contiguous_kv import Arena, native_contiguous


def elapsed(fn, n=8):
    for _ in range(2):
        fn()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(n):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / n


def byte_gate(seed, results):
    torch.manual_seed(seed)
    for page in (1, 16, 128):
        for batch, capacity in ((1, 257), (3, 533), (8, 193)):
            pages = (capacity + page - 1) // page
            cap = pages * page
            physical_pages = batch * pages + 7
            # Every FP8 representation, including signed zero and NaN, is copied.
            count = physical_pages * page * 2 * 64
            key_u = (torch.arange(count, device='cuda', dtype=torch.int64) % 256).to(torch.uint8)
            key = key_u.reshape(physical_pages, page, 2, 64).view(torch.float8_e4m3fn)
            value = (key_u ^ 173).reshape_as(key_u).reshape(key.shape).view(torch.float8_e4m3fn)
            lengths = torch.tensor([max(0, cap - i * 31 - 3) for i in range(batch)],
                                   dtype=torch.int32, device='cuda')
            # Strided table rows, shared prefix pages, and poison beyond lengths.
            base = torch.full((batch, pages + 3), -1234567, dtype=torch.int32, device='cuda')
            table = base[:, :pages]
            table.copy_(torch.randint(0, physical_pages, table.shape, device='cuda', dtype=torch.int32))
            if batch > 1:
                table[:, 0] = 2
            for b, length in enumerate(lengths.cpu().tolist()):
                table[b, (length + page - 1) // page:] = -1234567
            arena = Arena()
            added = arena.reserve(key, value, table, lengths)
            out_k, out_v = arena(key, value, table, lengths)
            expect_k, expect_v = torch.zeros_like(out_k.view(torch.uint8)), torch.zeros_like(out_v.view(torch.uint8))
            for b, length in enumerate(lengths.cpu().tolist()):
                rows = torch.arange(length, device='cuda')
                ids = table[b, rows // page].long() * page + rows % page
                expect_k[b, :length] = key.view(torch.uint8).reshape(-1, 2, 64)[ids]
                expect_v[b, :length] = value.view(torch.uint8).reshape(-1, 2, 64)[ids]
            exact = torch.equal(out_k.view(torch.uint8), expect_k) and torch.equal(out_v.view(torch.uint8), expect_v)
            # Capture stable storage and mutate all inputs/lengths/page IDs before replay.
            stream = torch.cuda.current_stream()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_k, graph_v = arena(key, value, table, lengths)
            key.view(torch.uint8).bitwise_xor_(37)
            value.view(torch.uint8).bitwise_xor_(91)
            lengths.sub_(1).clamp_(min=0)
            table[:, 0] = 1
            graph.replay()
            replay_exact = True
            for b, length in enumerate(lengths.cpu().tolist()):
                rows = torch.arange(length, device='cuda')
                ids = table[b, rows // page].long() * page + rows % page
                replay_exact &= torch.equal(graph_k[b, :length].view(torch.uint8), key.view(torch.uint8).reshape(-1, 2, 64)[ids])
                replay_exact &= torch.equal(graph_v[b, :length].view(torch.uint8), value.view(torch.uint8).reshape(-1, 2, 64)[ids])
                replay_exact &= bool((graph_k[b, length:].view(torch.uint8) == 0).all())
            blocked = False
            try:
                arena.reserve(key, value, table, lengths)
            except RuntimeError:
                blocked = True
            results.append(dict(page=page, batch=batch, capacity=cap, extra_bytes=added,
                                exact=bool(exact), graph_changed_exact=bool(replay_exact),
                                storage_replacement_blocked=blocked))
            assert exact and replay_exact and blocked
    return results


def attention_gate(seed, results):
    torch.manual_seed(seed)
    # Full attention only: SM86 native SWA origin bug is unrelated and excluded.
    configs = [([17, 3], [127, 43], 0), ([17, 3], [127, 43], 11),
               ([128, 63], [512, 129], 0), ([753], [4097], 15),
               ([384, 129, 3], [8192, 1025, 37], 0)]
    for qs, ks, padded in configs:
        batch, capacity = len(qs), max(ks)
        table = torch.randperm(batch * capacity, device='cuda', dtype=torch.int64).reshape(batch, capacity).int()
        lengths = torch.tensor(ks, dtype=torch.int32, device='cuda')
        for b, length in enumerate(ks):
            table[b, length:] = -1234567
        key = torch.randn(batch * capacity, 1, 2, 64, device='cuda', dtype=torch.bfloat16)
        value = torch.randn_like(key)
        q = torch.randn(sum(qs) + padded, 16, 64, device='cuda', dtype=torch.bfloat16)
        cuq = torch.tensor([0] + list(torch.tensor(qs).cumsum(0).tolist()), device='cuda', dtype=torch.int32)
        cuk = torch.tensor([0] + list(torch.tensor(ks).cumsum(0).tolist()), device='cuda', dtype=torch.int32)
        arena = Arena()
        added = arena.reserve(key, value, table, lengths)
        arena(key, value, table, lengths)
        for split in (0, 1, 3):
            args = dict(cu_seqlens_q=cuq, cu_seqlens_k_new=cuk, max_seqlen_q=max(qs),
                        causal=True, window_size=(-1, -1), num_splits=split,
                        softmax_scale=0.125, return_softmax_lse=True)
            def native():
                return flash_attn_with_kvcache(q, key, value, cache_seqlens=lengths,
                                              page_table=table, pack_gqa=True, **args)
            def candidate():
                return native_contiguous(flash_attn_with_kvcache, q, key, value,
                                         table, lengths, arena, **args)
            a, b = native(), candidate()
            real = sum(qs)
            exact = torch.equal(a[0][:real], b[0][:real])
            maxerr = (a[0][:real].float() - b[0][:real].float()).abs().max().item()
            # LSE native layout includes physical padded q slots. Compare real ones.
            lse_exact = torch.equal(a[1][..., :real], b[1][..., :real])
            changed_exact = True
            for _ in range(2):
                q.neg_().add_(torch.randn_like(q), alpha=0.125)
                key.neg_(); value.neg_()
                a, b = native(), candidate()
                changed_exact &= torch.equal(a[0][:real], b[0][:real])
            native_us = elapsed(native, 4)
            packed_us = elapsed(candidate, 4)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
                graph_result = candidate()
            q.mul_(-0.75); key.mul_(-0.5); value.add_(0.0625)
            graph.replay()
            reference = native()
            graph_exact = torch.equal(reference[0][:real], graph_result[0][:real])
            results.append(dict(qs=qs, ks=ks, padded=padded, splits=split, exact=bool(exact),
                                lse_exact=bool(lse_exact), max_abs_error=maxerr,
                                changed_exact=bool(changed_exact), graph_exact=bool(graph_exact),
                                native_us=native_us, packed_inclusive_us=packed_us,
                                ratio=native_us/packed_us, arena_bytes=added))
            assert exact and changed_exact and graph_exact
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=26011982)
    args = parser.parse_args()
    report = dict(seed=args.seed, device=torch.cuda.get_device_name(),
                  capability=list(torch.cuda.get_device_capability()),
                  source_sha256=hashlib.sha256(Path(__file__).with_name('contiguous_kv.py').read_bytes()).hexdigest(),
                  limitation='FP8 byte storage only; native attention is BF16 SM86. No Hopper FP8 parity or speed claim.')
    try:
        report['bytes'], report['attention'] = [], []
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            byte_gate(args.seed, report['bytes'])
            attention_gate(args.seed, report['attention'])
        report['passed'] = True
    except Exception as exc:
        report['passed'] = False
        report['error'] = repr(exc)
        raise
    finally:
        torch.cuda.synchronize()
        report['peak_allocated'] = torch.cuda.max_memory_allocated()
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
