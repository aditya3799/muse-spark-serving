"""Actual Triton bridge + graph replay test against independent CPU packers."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
from wgs_bridge import convert
import forward_oracles as oracle

here = Path(__file__).resolve().parent
root = here.parents[1]
source = root/'research_2026-11-18_target_moe/native_package_02/hopper_pack.py'
spec = importlib.util.spec_from_file_location('incumbent_wgs_pack', source)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
started = time.time()
report = dict(complete=False, device=torch.cuda.get_device_name(), rows=[],
              bridge_sha256=hashlib.sha256((here/'wgs_bridge.py').read_bytes()).hexdigest(),
              incumbent_pack_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
# One reusable 36 MiB expert-pair buffer; canaries outside the current stage.
storage = torch.full((2*8192*2304+128,), 0xA5, device='cuda', dtype=torch.uint8)
for seed in (930211, 930223):
    rng = np.random.default_rng(seed)
    for n in (8192,4096):
        e = 2
        q = rng.integers(0,16,(e,n,4096),dtype=np.uint8)
        sf = rng.integers(0,256,(e,n,256),dtype=np.uint8)
        q[0,0,:16]=np.arange(16,dtype=np.uint8);sf[0,0]=np.arange(256,dtype=np.uint8)
        raw = torch.from_numpy(q[...,::2] | (q[...,1::2]<<4)).cuda()
        scales = torch.from_numpy(sf).cuda().view(torch.float8_e4m3fn)
        packed = module.pack_experts(raw,scales,n==8192)
        expected_wgs=oracle.wgs_forward(q,sf,n==8192)
        assert np.array_equal(packed.cpu().numpy(), expected_wgs)
        del expected_wgs,raw,scales
        for layout in ('normal','humming_mma','humming_wgmma'):
            total=e*n*2304
            tail=storage[64+total:].clone()
            def call(): return convert(packed,storage[64:-64],n,gate_up=n==8192,layout=layout)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call();call();stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): a,b=call()
            torch.cuda.current_stream().wait_stream(stream)
            for changed in (False,True):
                if changed:
                    q[1,n-1,4095]^=8;sf[0,17,255]^=128
                    raw=torch.from_numpy(q[...,::2]|(q[...,1::2]<<4)).cuda()
                    scales=torch.from_numpy(sf).cuda().view(torch.float8_e4m3fn)
                    packed.copy_(module.pack_experts(raw,scales,n==8192))
                    del raw,scales
                graph.replay();torch.cuda.synchronize()
                ga,gb=oracle.normal_forward(q,sf) if layout=='normal' else oracle.humming_forward(q,sf,layout=='humming_wgmma')
                assert np.array_equal(a.cpu().numpy().view(np.uint8).reshape(-1),ga.view(np.uint8).reshape(-1)),(seed,n,layout,'codes',changed)
                assert np.array_equal(b.cpu().numpy().view(np.uint8).reshape(-1),gb.view(np.uint8).reshape(-1)),(seed,n,layout,'scales',changed)
                assert bool(torch.all(storage[:64]==0xA5))
                assert torch.equal(storage[64+total:],tail)
                row=dict(seed=seed,n=n,experts=e,layout=layout,changed_source=changed,
                         graph_replay=True,exact=True,nibbles=q.size,scale_bytes=sf.size,canaries=True)
                report['rows'].append(row);print(json.dumps(row),flush=True)
                del ga,gb
            del graph,a,b,tail
        del q,sf,packed
# Rejected API calls must fail before a CUDA launch.
w=torch.empty((4096*2304,),device='cuda',dtype=torch.uint8)
s=torch.empty_like(w)
bad=[lambda:convert(w,w,4096,gate_up=False,layout='normal'),
     lambda:convert(w,s[:-1],4096,gate_up=False,layout='normal'),
     lambda:convert(w,s,4096,gate_up=True,layout='normal'),
     lambda:convert(w,s,4096,gate_up=False,layout='unknown'),
     lambda:convert(w[::2],s,4096,gate_up=False,layout='normal'),
     lambda:convert(w.cpu(),s,4096,gate_up=False,layout='normal')]
for fn in bad:
    try:fn()
    except ValueError:pass
    else:raise AssertionError('invalid input accepted')
report.update(complete=True,negative_guards=len(bad),elapsed_seconds=time.time()-started,
              historical_peak_allocated=torch.cuda.max_memory_allocated(),
              historical_peak_reserved=torch.cuda.max_memory_reserved())
(here/'bridge_gpu.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
