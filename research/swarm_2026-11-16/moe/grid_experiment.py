"""Isolated Marlin persistent-grid experiment; never changes the serving checkout."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, shutil, statistics, sys, time
from pathlib import Path
from types import MethodType, SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'optimization_marlin_tiles_2026-11-15'


def build(root, dst, cap):
    if dst.exists():
        manifest=json.loads((dst/'manifest.json').read_text())
        cu=dst/'gemm/marlin_moe/moe_wna16_marlin.cuh'
        assert manifest['cap']==cap and manifest['modified_source_sha256']==hashlib.sha256(cu.read_bytes()).hexdigest()
        return dst/'runtime.py'
    dst.mkdir(parents=True, exist_ok=False)
    rel = 'python/sglang/kernels/jit/csrc/gemm'
    for part in ('marlin', 'marlin_moe'):
        shutil.copytree(root / rel / part, dst / 'gemm' / part)
    cu = dst / 'gemm/marlin_moe/moe_wna16_marlin.cuh'
    original = cu.read_text()
    old = '  return exec_cfg;\n}'
    assert original.count(old) == 1
    new = ('  // Experiment: bound persistent split-K contributors, preserving the selected\n'
           '  // tile, original weight representation, and FP32 reduction arithmetic.\n'
           f'  exec_cfg.blocks_per_sm = min(exec_cfg.blocks_per_sm, {cap});\n'
           '  return exec_cfg;\n}')
    cu.write_text(original.replace(old, new))
    py = (root / 'python/sglang/kernels/ops/moe/moe_wna16_marlin.py').read_text()
    src = 'cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],'
    assert py.count(src) == 1
    py = py.replace(src, f'cuda_files=[{str(cu)!r}],')
    runtime = dst / 'runtime.py'
    runtime.write_text(py)
    manifest = {'cap':cap,'base_source_sha256':hashlib.sha256(original.encode()).hexdigest(),
                'modified_source_sha256':hashlib.sha256(cu.read_bytes()).hexdigest(),
                'active_checkout_modified':False,'precision_changed':False,
                'weight_layout_changed':False,'gpu_scope':'single local shard'}
    (dst / 'manifest.json').write_text(json.dumps(manifest,indent=2))
    return runtime


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def main():
    p=argparse.ArgumentParser();p.add_argument('--sglang-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--compile-only',action='store_true')
    p.add_argument('--tokens',type=int,nargs='+',default=[8,32,128,256,512]);p.add_argument('--samples',type=int,default=5);p.add_argument('--caps',type=int,nargs='+',default=[1,2])
    p.add_argument('--hidden',type=int,default=4096);p.add_argument('--experts',type=int,default=32)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    paths={cap:build(args.sglang_root,args.out/f'cap{cap}',cap) for cap in args.caps}
    if args.prepare_only:return
    sys.path.insert(0,str(OLD))
    from kernel_runtime import bootstrap
    runtime=bootstrap(args.sglang_root)
    import torch
    import sglang.kernels.ops.moe.moe_wna16_marlin as marlin
    from marlin_bench import make_layer,load_method,capture,event_sample
    from tile_study import make_routes
    from sglang.kernels.ops.moe.muse_spark_post_expert_finalize_triton import fused_post_expert_norm_weighted_sum
    variants={'baseline':marlin.moe_wna16_marlin_gemm}
    variants.update({f'cap{k}':load(v,f'marlin_cap{k}').moe_wna16_marlin_gemm for k,v in paths.items()})
    if args.compile_only:
        logs=[]
        for k,path in paths.items():
            mod=load(path,f'compile_cap{k}')
            started=time.time();compiled=mod._jit_moe_wna16_marlin_module(torch.bfloat16,True,False,False,False,False)
            logs.append({'variant':f'cap{k}','elapsed_s':time.time()-started,'source_sha256':hashlib.sha256((path.parent/'gemm/marlin_moe/moe_wna16_marlin.cuh').read_bytes()).hexdigest(),'module':str(compiled)})
            print(json.dumps(logs[-1]),flush=True)
            (args.out/'compile.json').write_text(json.dumps(logs,indent=2))
        return
    torch.manual_seed(101)
    cfg=SimpleNamespace(sglang_root=args.sglang_root,hidden=args.hidden,intermediate=args.hidden,
                        local_experts=args.experts,global_experts=args.experts*8,rank=3)
    def routes(tokens,pattern,seed):
        values=make_routes(tokens,8,args.experts*8,8,pattern,seed)
        return [[(v+3*args.experts)%(8*args.experts) if v>=0 else -1 for v in row] for row in values]
    layer=make_layer(cfg,torch)
    method=MethodType(load_method(args.sglang_root,'baseline',torch)[0],layer)
    weight=torch.linspace(.5,1.5,args.hidden,device='cuda',dtype=torch.bfloat16)
    flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
    report={'runtime':runtime,'gpu':torch.cuda.get_device_name(),'scope':'component only; no serving claim','rows':[]}
    report_path=args.out/'results.json'
    for tokens in args.tokens:
      for pattern in ('uniform','hot_rank','hot_expert','all_remote'):
        x=torch.randn((tokens,args.hidden),device='cuda',dtype=torch.bfloat16)/20
        ids=torch.tensor(routes(tokens,pattern,101),device='cuda',dtype=torch.int32)
        scores=torch.rand((tokens,8),device='cuda');scores/=scores.sum(-1,keepdim=True)
        def complete():
            values,local_ids=method(x,scores,ids)
            return fused_post_expert_norm_weighted_sum(values,scores,norm_weight=weight,eps=1e-6,weight_first=True,slot_expert_ids=local_ids)
        graphs={}
        for name,fn in variants.items():
            marlin.moe_wna16_marlin_gemm=fn
            graphs[name]=capture(torch,complete)
        if not report.get('loaded_binary_provenance'):
            mapped={}
            for entry in Path('/proc/self/maps').read_text().splitlines():
                if 'sgl_kernel_jit_moe_wna16_marlin' in entry and '.so' in entry:
                    binary=Path(entry.split()[-1])
                    if binary.is_file():
                        mapped[str(binary)]=hashlib.sha256(binary.read_bytes()).hexdigest()
            report['loaded_binary_provenance']=mapped
        row={'tokens':tokens,'pattern':pattern,'validation':[],'samples_us':{k:[] for k in graphs}}
        for seed in (102,103,104):
            torch.manual_seed(seed);x.copy_(torch.randn_like(x)/20)
            # Mutate routing to test replay input dependence while keeping the pattern.
            ids.copy_(torch.tensor(routes(tokens,pattern,seed),device='cuda',dtype=torch.int32))
            scores.copy_(torch.rand_like(scores));scores.div_(scores.sum(-1,keepdim=True))
            for g,out in graphs.values():g.replay()
            torch.cuda.synchronize()
            ref=graphs['baseline'][1]
            for name,(g,out) in graphs.items():
                diff=out.float()-ref.float();rel=(diff.norm()/ref.float().norm().clamp_min(1e-20)).item()
                maximum=diff.abs().max().item();finite=bool(torch.isfinite(out).all())
                valid=finite and rel<=.0005 and maximum<=.015625
                row['validation'].append({'seed':seed,'variant':name,'relative_l2':rel,'max_abs':maximum,'passed':valid})
        for rep in range(args.samples):
            names=list(graphs) if rep%2==0 else list(reversed(graphs))
            for name in names:
                flush.zero_();row['samples_us'][name].append(event_sample(torch,graphs[name][0].replay,1))
        row['median_us']={k:statistics.median(v) for k,v in row['samples_us'].items()}
        row['speedups']={k:row['median_us']['baseline']/v for k,v in row['median_us'].items()}
        row['passed']=all(v['passed'] for v in row['validation'])
        report['rows'].append(row);report_path.write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ('validation','samples_us')}),flush=True)
        del graphs
    marlin.moe_wna16_marlin_gemm=variants['baseline']
    report['passed']=all(r['passed'] for r in report['rows']);report_path.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()



