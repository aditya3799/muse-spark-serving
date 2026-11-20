"""Persistent-grid correctness plus the frozen reviewed metadata timing harness."""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import metadata_probe as base

P = Path(__file__).resolve().parent
NEW = P / 'persistent_metadata/candidate_sources/python/sglang/kernels/ops/attention/metadata.py'
PINS = dict(base.PINS)
del PINS[base.NEW]
PINS[NEW] = '2125af76349ba0ceda9f466ab0a1ab6589e2ad23abfd96547b7ddf20ee60618e'
PINS[P / 'metadata_probe.py'] = '43b08e3b818681be17782cd34f1f191971a5daca235b8207dc98492f9495997c'

def scalar_edge_cases(torch, old, new, save_progress):
    checked = 0
    for width in (0, 1):
     for batch in (1, 3):
      for page in (1, 16, 256):
       for skip in (False, True):
        for tail in (False, True):
         seq = torch.zeros(batch, device='cuda', dtype=torch.int32)
         table = torch.zeros((batch, max(1, width * page)), device='cuda', dtype=torch.int32)
         req = torch.arange(batch, device='cuda', dtype=torch.int32)
         inputs = dict(req_to_token=table, req_pool_indices=req, max_seq_pages=width, seq_lens=seq,
                       seq_len_delta=0, page_size=page, skip_page_table=skip, zero_tail=tail)
         outputs = [dict(cache_seqlens_int32=torch.full((batch,), -19, device='cuda', dtype=torch.int32),
                         cu_seqlens_k=torch.full((batch + 1,), -19, device='cuda', dtype=torch.int32),
                         page_table=torch.full((batch, width), -19, device='cuda', dtype=torch.int32)) for _ in range(2)]
         for _ in range(3):
            old.normal_decode_set_metadata(**outputs[0], **inputs)
            new.normal_decode_set_metadata(**outputs[1], **inputs)
         torch.cuda.synchronize()
         graph = torch.cuda.CUDAGraph()
         with torch.cuda.graph(graph): new.normal_decode_set_metadata(**outputs[1], **inputs)
         for length in (0, width * page, 0):
            seq.fill_(length)
            old.normal_decode_set_metadata(**outputs[0], **inputs); graph.replay(); torch.cuda.synchronize()
            for name in outputs[0]: assert torch.equal(outputs[0][name], outputs[1][name]), ('scalar-edge', width, batch, page, skip, tail, name)
            checked += 1
         del graph, outputs
         save_progress(dict(status='scalar_edges_running', passed=False, scalar_edge_replays=checked))
    return checked

def correctness(torch, old, new, save_progress):
    cases = 0
    save_progress(dict(status='running', passed=False, changed_input_graph_cases=0))
    for page in (1, 16, 256):
      for batch in (1, 3, 17):
       width = 257
       rows = batch + 2
       stride_width = width * page
       table = torch.arange(rows * stride_width, device='cuda', dtype=torch.int32).view(rows, stride_width)
       req = torch.arange(batch, device='cuda', dtype=torch.int32)
       seq_storage = torch.zeros(batch * 2, device='cuda', dtype=torch.int64)
       seq = seq_storage[::2]
       full_map = torch.arange(rows * width, device='cuda', dtype=torch.int32).flip(0)
       full_map[::7] = -1
       swa_map = full_map.clone()
       static_map = torch.arange(rows * stride_width, device='cuda', dtype=torch.int32).flip(0)
       for mapping in ('plain', 'static_swa', 'translated_swa', 'translated_full', 'skip'):
        for zero_tail in (False, True):
         args = dict(req_to_token=table, req_pool_indices=req, max_seq_pages=width,
                     seq_lens=seq, seq_len_delta=0, page_size=page,
                     full_to_swa_index_mapping=static_map if mapping == 'static_swa' else None,
                     full_v2p=full_map if mapping.startswith('translated') else None,
                     swa_v2p=swa_map if mapping == 'translated_swa' else None,
                     skip_page_table=mapping == 'skip', zero_tail=zero_tail)
         buffers = []
         stores = []
         for _ in range(2):
            cache = torch.full((batch * 2,), -19, device='cuda', dtype=torch.int32)
            cu = torch.full(((batch + 1) * 2,), -19, device='cuda', dtype=torch.int32)
            pages = torch.full((batch, width * 2), -19, device='cuda', dtype=torch.int32)
            swa = torch.full_like(pages, -19) if 'swa' in mapping else None
            stores.append((cache, cu, pages, swa))
            buffers.append(dict(cache_seqlens_int32=cache[::2], cu_seqlens_k=cu[::2],
                                page_table=pages[:, ::2], swa_page_table=None if swa is None else swa[:, ::2]))
         seq.fill_(page)
         for _ in range(3):
            old.normal_decode_set_metadata(**buffers[0], **args)
            new.normal_decode_set_metadata(**buffers[1], **args)
         torch.cuda.synchronize()
         graph = torch.cuda.CUDAGraph()
         with torch.cuda.graph(graph): new.normal_decode_set_metadata(**buffers[1], **args)
         for step, length in enumerate((0, 1, page, 128 * page - 1, 128 * page + 1, width * page, 0)):
            seq.copy_(torch.clamp(torch.arange(batch, device='cuda', dtype=torch.int64) + length, max=width * page))
            req.copy_((torch.arange(batch, device='cuda', dtype=torch.int32) + step) % rows)
            # Change all mapping arrays at stable addresses; retain -1 sink cases.
            full_map.copy_(full_map.roll(1)); swa_map.copy_(swa_map.roll(-1)); static_map.copy_(static_map.roll(1))
            old.normal_decode_set_metadata(**buffers[0], **args)
            graph.replay()
            torch.cuda.synchronize()
            for a, b in zip(stores[0], stores[1]):
                if a is not None: assert torch.equal(a, b), (page, batch, mapping, zero_tail, step)
            # Independent scalar page-id oracle for every semantically live page.
            for row in range(batch):
                live = (int(seq[row]) + page - 1) // page
                if mapping == 'skip':
                    assert bool((buffers[1]['page_table'][row] == -19).all())
                    continue
                token = table[int(req[row]), :live * page:page]
                full = token // page
                if args['full_v2p'] is not None: full = full_map[full.long()].clamp_min(0)
                assert torch.equal(buffers[1]['page_table'][row, :live], full)
                if mapping == 'static_swa': wanted = static_map[token.long()] // page
                elif mapping == 'translated_swa': wanted = swa_map[(token // page).long()].clamp_min(0)
                else: wanted = None
                if wanted is not None: assert torch.equal(buffers[1]['swa_page_table'][row, :live], wanted)
                if zero_tail:
                    assert bool((buffers[1]['page_table'][row, live:] == 0).all())
                    if wanted is not None: assert bool((buffers[1]['swa_page_table'][row, live:] == 0).all())
            cases += 1
            if cases % 7 == 0:
                save_progress(dict(status='running', passed=False, changed_input_graph_cases=cases,
                                   last_completed_geometry=[page, batch, mapping, zero_tail]))
         del graph, buffers, stores
         torch.cuda.synchronize()
    return dict(passed=True, changed_input_graph_cases=cases,
                coverage='page1/16/256, B1/3/17, noncontiguous lens/output strides, plain/staticSWA/unifiedfull+SWA/fullonly/skip, ZERO_TAIL false+true, long-short-live0 transitions, mutated maps and request IDs')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args, _ = parser.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path, digest in PINS.items(): assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    import torch
    old = base.load(base.OLD, 'persistent_old_metadata')
    new = base.load(NEW, 'persistent_new_metadata')
    def save_progress(proof):
        proof['candidate_sha256'] = PINS[NEW]
        (args.out / 'correctness.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    edge_count = scalar_edge_cases(torch, old, new, save_progress)
    (args.out / 'scalar_edges.json').write_text(json.dumps(dict(passed=True, replays=edge_count, candidate_sha256=PINS[NEW]), indent=2) + '\n', encoding='utf-8')
    proof = correctness(torch, old, new, save_progress)
    proof['scalar_edge_replays'] = edge_count
    proof['candidate_sha256'] = PINS[NEW]
    (args.out / 'correctness.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(proof), flush=True)
    base.NEW = NEW
    base.PINS = PINS
    base.main()
    report_path = args.out / 'results.json'
    report = json.loads(report_path.read_text())
    report['persistent_correctness'] = proof
    report['wrapper_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['variant_labels'] = {'AE': 'original eager', 'AG': 'original glue', 'PE': 'persistent eager', 'PG': 'persistent glue'}
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__': main()
