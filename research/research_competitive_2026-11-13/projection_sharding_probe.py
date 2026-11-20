"""Research probe: replicated BF16 projection versus output-column sharding.

Run on the target node (does not launch serving or submit a judge job):
  torchrun --standalone --nproc-per-node=8 projection_sharding_probe.py --output results.json

The candidate uses generic NCCL all-gather plus a layout conversion, not SGLang's
multicast gather. Results establish a portable baseline, not the latter's speed.
The CPU self-test checks matrix partition/reconstruction only.
"""
import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


def reconstruct(gathered, world, tokens, columns):
    return gathered.view(world, tokens, columns).permute(1, 0, 2).reshape(tokens, world * columns)


def self_test():
    torch.manual_seed(71)
    count = 0
    largest = 0.0
    for world in (2, 4, 8):
        for tokens in (1, 3, 8, 17, 32):
            x = torch.randn(tokens, 128, dtype=torch.float64)
            w = torch.randn(256, 128, dtype=torch.float64)
            expected = F.linear(x, w)
            pieces = [F.linear(x, part) for part in w.chunk(world, dim=0)]
            actual = reconstruct(torch.cat(pieces), world, tokens, 256 // world)
            error = (actual - expected).abs().max().item()
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            largest = max(largest, error)
            count += 1
    return {"kind": "CPU float64 partition/reconstruction test", "cases": count,
            "maximum_absolute_difference": largest,
            "validates": "row ordering and mathematical partition only; no distributed or GPU timing"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', default='projection_probe_results.json')
    parser.add_argument('--tokens', default='1,3,8,15,32,63,128,255,256,512,1024,4096,8192,16384')
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--inner', type=int, default=5)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
        return
    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU and torchrun distributed launch are required.')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise SystemExit('This research protocol requires the actual 8-rank topology.')
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit('Use Hopper for target-device evidence; laptop timing is not comparable.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    records = []
    with torch.inference_mode():
        for n, k in ((4096, 8192), (8192, 4096)):
            torch.manual_seed(7100 + n)
            w = torch.randn(n, k, device='cuda', dtype=torch.bfloat16) * 0.01
            # Broadcast ensures truly identical weights and inputs across ranks.
            dist.broadcast(w, src=0)
            local_w = w[rank * (n // world):(rank + 1) * (n // world)]
            for m in map(int, args.tokens.split(',')):
                x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
                dist.broadcast(x, src=0)
                local_out = torch.empty(m, n // world, device='cuda', dtype=torch.bfloat16)
                gathered = torch.empty(world * m, n // world, device='cuda', dtype=torch.bfloat16)
                full_out = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
                candidate_out = torch.empty_like(full_out)

                def reference(x=x, w=w, full_out=full_out):
                    torch.mm(x, w.t(), out=full_out)
                    return full_out

                def candidate(x=x, local_w=local_w, local_out=local_out, gathered=gathered, candidate_out=candidate_out, m=m, world=world, n=n):
                    torch.mm(x, local_w.t(), out=local_out)
                    dist.all_gather_into_tensor(gathered, local_out)
                    candidate_out.view(m, world, n // world).copy_(
                        gathered.view(world, m, n // world).permute(1, 0, 2))
                    return candidate_out

                for _ in range(5):
                    reference()
                    candidate()
                torch.cuda.synchronize()
                expected, actual = reference().clone(), candidate().clone()
                error = actual.float() - expected.float()
                finite = bool(torch.isfinite(actual).all().item())
                correctness = {
                    'finite': finite,
                    'max_absolute_difference': error.abs().max().item(),
                    'relative_l2': (error.norm() / expected.float().norm().clamp_min(1e-30)).item(),
                    'bitwise_equal': bool(torch.equal(actual, expected)),
                }
                stats = torch.tensor([float(not finite), correctness['max_absolute_difference'],
                                      correctness['relative_l2'], float(not correctness['bitwise_equal'])],
                                     device='cuda', dtype=torch.float64)
                dist.all_reduce(stats, op=dist.ReduceOp.MAX)
                worst = stats.tolist()
                correctness = {'all_ranks_finite': worst[0] == 0,
                               'all_ranks_max_absolute_difference': worst[1],
                               'all_ranks_max_relative_l2': worst[2],
                               'all_ranks_bitwise_equal': worst[3] == 0}
                if worst[0] != 0:
                    raise RuntimeError('Nonfinite projection output on at least one rank')
                del expected, actual, error
                samples = {'reference': [], 'candidate': []}
                # Alternate paired order. Synchronize rank starts outside timed events;
                # take the maximum rank time outside the measured interval.
                for rep in range(args.repetitions):
                    order = [('reference', reference), ('candidate', candidate)]
                    if rep % 2:
                        order.reverse()
                    for label, op in order:
                        dist.barrier()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(args.inner):
                            op()
                        end.record()
                        end.synchronize()
                        elapsed = torch.tensor(start.elapsed_time(end) / args.inner,
                                               device='cuda', dtype=torch.float64)
                        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                        samples[label].append(elapsed.item())
                med = {key: statistics.median(value) for key, value in samples.items()}
                rec = {'tokens': m, 'output_features': n, 'input_features': k,
                       'correctness_all_ranks': correctness,
                       'max_rank_median_ms': med,
                       'speedup': med['reference'] / med['candidate'],
                       'raw_max_rank_ms': samples,
                       'candidate_extra_buffer_bytes': (local_out.numel() + gathered.numel()) * 2}
                records.append(rec)
                if rank == 0:
                    print(json.dumps({k: v for k, v in rec.items() if k != 'raw_max_rank_ms'}), flush=True)
                del reference, candidate, order, op, x, local_out, gathered, full_out, candidate_out
            del w, local_w
    if rank == 0:
        result = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                  'world': world, 'cuda_graphs': False,
                  'scope': 'Eager synthetic dense projection + generic NCCL. Not serving, not custom multicast.',
                  'followup': 'Validate on real activations, all ranks, captured SGLang multicast path and serving.',
                  'records': records}
        Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

