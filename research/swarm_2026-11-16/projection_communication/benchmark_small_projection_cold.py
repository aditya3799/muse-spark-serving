"""Cold-pressure component timing of the frozen round2 local projection.

Audit mode is CPU-only. GPU mode requires an exclusive slot and an external
process timeout. Results are compute components, never serving speedup.
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROUND2 = HERE.parent / 'coordination/round2_projection_attention'
REL_HELPER = 'python/sglang/srt/layers/communication/muse_spark_gemm_ag.py'
REL_MODEL = 'python/sglang/srt/models/muse_spark.py'
EXPECTED = {
    REL_HELPER: 'd6b60d95757f6fadae7946885fc5d373d4eae6612ebb48595b12bce10bcbc94e',
    REL_MODEL: '255b2ce03ac0ecc500e230c74b8f3966144bf7e8009b99bd77d77dd9a470f0c6',
    'candidate.diff': 'b9004309c2a45568e901ce5ab86379c65bad3af41f0ed1aa3f7f5a4971e4c9c4',
}
SHAPES = [(m,k,n) for k,n in ((8192,4096),(4096,8192)) for m in (1,8,16,32,64,128,256,511)]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    hashes = {}
    for rel, expected in EXPECTED.items():
        path = ROUND2 / ('candidate_sources/' + rel if rel.startswith('python/') else rel)
        hashes[rel] = digest(path)
        if hashes[rel] != expected:
            raise RuntimeError('Frozen source mismatch: ' + rel)
        if rel.startswith('python/'):
            ast.parse(path.read_bytes())
            if digest(HERE / 'candidate_small_multicast' / rel) != expected:
                raise RuntimeError('Original validated export mismatch: ' + rel)
    prior_path = HERE / 'small_gpu_numerical.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    if not prior['passed'] or prior['helper_sha256'] != EXPECTED[REL_HELPER]:
        raise RuntimeError('Numeric evidence does not cover exact helper')
    for m,k,n in SHAPES:
        row = next(r for r in prior['rows'] if (r['m'],r['k'],r['n']) == (m,k,n))
        if len(row['errors']) != 6 or not all(e['passed'] for e in row['errors']):
            raise RuntimeError('Incomplete numeric evidence')
    return dict(hashes=hashes,harness_sha256=digest(Path(__file__)),
                numeric_evidence_sha256=digest(prior_path),numeric_cases=96,
                shapes=[dict(m=m,k=k,n=n,serving_branch_changed=not(k==4096 and n==8192 and m<=8)) for m,k,n in SHAPES])

def abba_orders(cycles):
    for cycle in range(cycles):
        yield ('A','B','B','A') if cycle % 2 == 0 else ('B','A','A','B')

def quality(medians, paired_ratios):
    aa_ratio = statistics.median(paired_ratios)
    return dict(aa_ratio=aa_ratio, aa_within_10_percent=abs(aa_ratio-1)<=0.10,
                aa_cycle_min=min(paired_ratios),aa_cycle_max=max(paired_ratios),
                usable=abs(aa_ratio-1)<=0.10 and all(abs(r-1)<=0.20 for r in paired_ratios),
                local_vs_full_ratio=medians['local_fp32']/medians['full'],
                compute_only_gather_budget_us=medians['full']-medians['local_fp32'])

def run(args, evidence):
    import torch
    import torch.nn.functional as F
    if not args.gpu_authorized:
        raise RuntimeError('GPU execution requires root-authorized exclusive slot')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    started=time.monotonic();deadline=started+args.max_seconds
    def bound():
        if time.monotonic()>deadline:
            raise TimeoutError('Cooperative runtime budget exceeded')
    tree=ast.parse((ROUND2/'candidate_sources'/REL_HELPER).read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='small_projection_local')
    scope={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROUND2/'candidate_sources'/REL_HELPER),'exec'),scope)
    local=scope['small_projection_local']
    props=torch.cuda.get_device_properties(torch.cuda.current_device())
    l2=getattr(props,'L2_cache_size',getattr(props,'l2_cache_size',None))
    evict_bytes=max(128*1024*1024,2*int(l2 or 0))
    out=HERE/('cold_timing_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+str(os.getpid()))
    out.mkdir(exist_ok=False)
    report=dict(evidence=evidence,gpu=props.name,capability=[props.major,props.minor],
        torch_version=torch.__version__,cuda_version=torch.version.cuda,weight_rotation=8,
        full_weight_rotation_bytes=8*64*1024*1024,local_weight_rotation_bytes=8*8*1024*1024,
        cache_pressure_bytes=evict_bytes,reported_l2_bytes=l2,
        cache_note='Read/write cache pressure before every arm, outside event interval. Cache residency is not directly measured.',
        baseline='torch.nn.functional.linear component; not exact Hopper stock GEMV/split-K dispatcher',
        scope='Exact local FP32 helper plus BF16 cast; no multicast, NCCL, model execution or serving speed claim.',
        timings_exclude='Tensor allocation, compilation, warmup, capture and cache pressure. CUDA graph captured allocation-dependent work remains included.',
        rows=[],status='running')
    report_path=out/'report.json'
    def save():report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save();print(json.dumps({'output':str(out),'status':'started'}),flush=True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    try:
      with torch.inference_mode():
        evict=torch.zeros(evict_bytes//4,device='cuda',dtype=torch.float32)
        # Allocate and initialize events before measurements.
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();end.record();end.synchronize()
        for k,n in ((8192,4096),(4096,8192)):
          if args.geometry=='pool' and k!=8192:continue
          if args.geometry=='unpool' and k!=4096:continue
          bound();torch.manual_seed(711)
          weights=[torch.randn(n,k,device='cuda',dtype=torch.bfloat16) for _ in range(8)]
          for shape_index,m in enumerate(args.tokens):
            bound();x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            comm=SimpleNamespace(world_size=8,rank=0)
            calls={'full':lambda:[F.linear(x,w) for w in weights],
                   'full_control':lambda:[F.linear(x,w) for w in weights],
                   'local_fp32':lambda:[local(comm,x,w) for w in weights]}
            graphs={};outputs={}
            # Each graph owns its pool, all are built before measurement.
            capture_order=list(calls) if shape_index%2 else list(reversed(calls))
            for name in capture_order:
              stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
              with torch.cuda.stream(stream):
                for _ in range(3):calls[name]()
              torch.cuda.current_stream().wait_stream(stream)
              graph=torch.cuda.CUDAGraph()
              with torch.cuda.graph(graph):value=calls[name]()
              graphs[name]=graph;outputs[name]=value
            for _ in range(3):
              for graph in graphs.values():graph.replay()
            torch.cuda.synchronize();bound()
            def arm(name):
              bound();evict.add_(1)
              start.record();graphs[name].replay();end.record();end.synchronize()
              return start.elapsed_time(end)*1000/len(weights)
            samples=[];aa_ratios=[];paired_ratios=[];paired_budgets=[]
            # Alternate real A/B and identical A/A pairs; each uses ABBA/BAAB.
            for cycle,order in enumerate(abba_orders(args.cycles)):
              pair_order=('AB','AA') if cycle%2==0 else ('AA','AB')
              for pair in pair_order:
                names={'A':'full','B':'local_fp32' if pair=='AB' else 'full_control'}
                values={key:[] for key in names}
                for position,key in enumerate(order):
                  us=arm(names[key]);values[key].append(us)
                  samples.append(dict(cycle=cycle,pair=pair,position=position,arm=key,name=names[key],us=us))
                ratio=statistics.mean(values['B'])/statistics.mean(values['A'])
                (paired_ratios if pair=='AB' else aa_ratios).append(ratio)
                if pair=='AB':paired_budgets.append(statistics.mean(values['A'])-statistics.mean(values['B']))
            med={name:statistics.median(s['us'] for s in samples if s['name']==name) for name in calls}
            row=dict(m=m,k=k,n=n,serving_branch_changed=not(k==4096 and n==8192 and m<=8),
                branch_note='Unchanged native unpool control' if k==4096 and n==8192 and m<=8 else 'New small-projection branch geometry',
                median_us=med,paired_local_full_median=statistics.median(paired_ratios),
                paired_compute_only_gather_budget_us=statistics.median(paired_budgets),
                quality=quality(med,aa_ratios),samples=samples,
                resident_allocated_bytes=torch.cuda.memory_allocated(),resident_reserved_bytes=torch.cuda.memory_reserved())
            report['rows'].append(row);save();print(json.dumps({key:row[key] for key in ('m','k','n','serving_branch_changed','median_us','quality')}),flush=True)
            # No graph or closure survives the shape; cleanup is outside timings.
            del graphs,outputs,calls,graph,value,x,stream
            torch.cuda.synchronize();torch.cuda.empty_cache()
          del weights
          torch.cuda.synchronize();torch.cuda.empty_cache()
        del evict
        torch.cuda.synchronize();torch.cuda.empty_cache()
      report['status']='complete'
      report['all_aa_controls_pass']=all(r['quality']['usable'] for r in report['rows'])
    except BaseException as exc:
      report['status']='failed';report['error']=repr(exc)
      raise
    finally:
      report['elapsed_seconds']=time.monotonic()-started;save()
      print(json.dumps({'status':report['status'],'report':str(report_path)}),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu-authorized',action='store_true')
    parser.add_argument('--cycles',type=int,default=8)
    parser.add_argument('--tokens',type=int,nargs='+',choices=(1,8,16,32,64,128,256,511),default=(1,8,16,32,64,128,256,511))
    parser.add_argument('--geometry',choices=('both','pool','unpool'),default='both')
    parser.add_argument('--max-seconds',type=int,default=540)
    args=parser.parse_args()
    if not 2<=args.cycles<=16 or not 30<=args.max_seconds<=540:
        parser.error('cycles must be2..16; max-seconds30..540')
    evidence=audit()
    if not args.gpu_authorized:
        print(json.dumps(dict(status='cpu_audit_passed',**evidence),indent=2));return
    run(args,evidence)

if __name__=='__main__':main()
