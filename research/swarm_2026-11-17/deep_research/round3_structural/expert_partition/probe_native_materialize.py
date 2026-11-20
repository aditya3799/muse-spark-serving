"""Bounded materializer-only correctness and matched best-Triton timing."""
from pathlib import Path
import argparse,gc,hashlib,json,random,statistics,sys,time
P=Path(__file__).resolve().parent;ROOT=P.parents[3];A=P.parent/'aux_speculation';sys.path.insert(0,str(ROOT/'swarm_2026-11-16/moe/humming_vendor'));sys.path.insert(0,str(A))
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);pa.add_argument('--samples',type=int,default=8);args=pa.parse_args();assert args.samples>=4 and args.samples%2==0;args.out.mkdir(parents=True,exist_ok=True)
 if (args.out/'results.json').exists():raise ValueError('Fresh result required')
 import torch,triton
 from packed_staged_ops import packed_dequant_kernel
 from native_materialize import NativeMaterializer
 r={'status':'started','scope':'Direct native CUDA materialization only, E1 physical packedNVFP4/FP8->quarterBF16; bestTriton256/8 reference. No fullhelper, genericE32 orH200 speed claim.','sources':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),P/'native_materialize.py',P/'native_materialize.cuh',A/'packed_staged_ops.py']},'checks':[],'kernels':[],'timings':[],'environment':'Independent user-authorized TTS service unchanged; timing provisional.'}
 def save():(args.out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
 def exact(a,b,**meta):
  af=a.flatten();bf=b.flatten();good=True;finite=True
  for lo in range(0,af.numel(),1<<20):
   av=af[lo:lo+(1<<20)];bv=bf[lo:lo+(1<<20)];good=good and torch.equal(av.view(torch.int16),bv.view(torch.int16));finite=finite and bool(torch.isfinite(av).all() and torch.isfinite(bv).all())
  row=dict(meta,bitwise=good,finite=finite,elements=af.numel());r['checks'].append(row);save();assert good and finite,row
 free,total=torch.cuda.mem_get_info();r['preflight']={'free':free,'estimated_additional':512*1024**2,'reserve':512*1024**2};save()
 if free<1024*1024**2:r['status']='HELD_MEMORY_RESERVE';save();return
 props=torch.cuda.get_device_properties();assert(props.major,props.minor)==(8,6)
 def reserve_check(phase):
  free_now,total_now=torch.cuda.mem_get_info();r.setdefault('memory_checks',[]).append({'phase':phase,'free':free_now,'allocated':torch.cuda.memory_allocated()});save()
  if free_now<512*1024**2:r['status']='HELD_MEMORY_RESERVE';save();raise RuntimeError('Memory reserve exhausted')
 def inputs(n,k,generation):
  torch.manual_seed(819+generation);w=torch.randint(-(1<<31),(1<<31)-1,(n*k//8,),device='cuda',dtype=torch.int32)
  b=torch.randint(0,256,(n*k//16,),device='cuda',dtype=torch.uint8);b=torch.where((b&127)==127,b-1,b)
  return w,b.view(torch.float8_e4m3fn)
 def compile_kernel(n,k,src,dst,loop,label):
  kernel=NativeMaterializer(n,k,source_transpose=src,target_transpose=dst,fixed_grid_loop=loop);dest=args.out/label;dest.mkdir(exist_ok=True);(dest/'kernel.cu').write_text(kernel.code);cb=Path(kernel.kernel_filename);(dest/'kernel.sm86.cubin').write_bytes(cb.read_bytes());logs=''
  for f in ['stdout.log','stderr.log','cmdline.json']:
   if (cb.parent/f).exists():(dest/f).write_bytes((cb.parent/f).read_bytes());logs+=(cb.parent/f).read_text(errors='replace')+'\n'
  r['kernels'].append({'label':label,'source_transpose':src,'target_transpose':dst,'fixed_grid_loop':loop,'blocks':kernel.blocks,'threads':kernel.threads,'source_sha256':kernel.source_sha256,'cubin_sha256':hashlib.sha256(cb.read_bytes()).hexdigest(),'resources':[x for x in logs.splitlines() if any(t in x for t in ['Used ','spill','stack frame'])]});save();return kernel
 # Small all-layout native gate precedes full allocations/timing.
 n=64;k=128;w,s=inputs(n,k,0);ref=torch.empty(n*k,device='cuda',dtype=torch.bfloat16);storage=torch.full((n*k+8,),37.,device='cuda',dtype=torch.bfloat16);out=storage[:n*k]
 for src in [False,True]:
  for dst in [False,True]:
   packed_dequant_kernel[(triton.cdiv(n*k,256),)](w.view(torch.uint8),s.view(torch.uint8),ref,n,k,src,dst,256,num_warps=8)
   for loop in [False,True]:
    kernel=compile_kernel(n,k,src,dst,loop,'small_%d_%d_%d'%(src,dst,loop));out.fill_(37.);kernel(w,s,out);torch.cuda.synchronize();exact(out,ref,scope='all_layout_small',source_transpose=src,target_transpose=dst,loop=loop);assert bool((storage[n*k:]==37).all())
 del w,s,ref,storage,out,kernel;gc.collect();torch.cuda.empty_cache();print('small_all_layout_gate_passed',flush=True)
 flush=torch.empty(64*1024**2,device='cuda',dtype=torch.uint8)
 def sample(graph,cold):
  if cold:flush.fill_(97)
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();graph.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
 for stage,n in [('w13',8192),('w2',4096)]:
  k=4096;w,s=inputs(n,k,0);ref=torch.empty(n*k,device='cuda',dtype=torch.bfloat16);out=torch.empty_like(ref);assert ref.data_ptr()!=out.data_ptr();graphs={};native={}
  def reference():return packed_dequant_kernel[(triton.cdiv(n*k,256),)](w.view(torch.uint8),s.view(torch.uint8),ref,n,k,False,False,256,num_warps=8)
  for loop in [False,True]:native[loop]=compile_kernel(n,k,False,False,loop,stage+('_fixed' if loop else '_tiled'))
  reserve_check(stage+'_allocated')
  calls={'reference':reference,'tiled':lambda:native[False](w,s,out),'fixed':lambda:native[True](w,s,out)}
  for label,call in calls.items():
   for _ in range(2):call()
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):call()
   graphs[label]=g
  for generation in range(3):
   if generation:
    nw,ns=inputs(n,k,generation);w.copy_(nw);s.copy_(ns);del nw,ns
   ref.fill_(float('nan'));graphs['reference'].replay();torch.cuda.synchronize()
   for label in ['tiled','fixed']:
    out.fill_(float('nan'));graphs[label].replay();torch.cuda.synchronize();exact(out,ref,stage=stage,generation=generation,variant=label)
  print(stage+'_full_gate_passed',flush=True)
  for label in ['tiled','fixed']:
   for cold in [False,True]:
    reserve_check(stage+'_'+label+'_timing')
    ga=graphs['reference'];gb=graphs[label];until=time.monotonic()+.3
    while time.monotonic()<until:sample(ga,cold);sample(gb,cold)
    values={x:[] for x in ['A1','B1','A2','B2']};rng=random.Random(n+int(cold)+int(label=='fixed'))
    for repeat in range(args.samples):
     arms=list(values);rng.shuffle(arms)
     for arm in arms:values[arm].append(sample(ga if arm.startswith('A') else gb,cold))
    med={x:statistics.median(y) for x,y in values.items()};ratio=lambda a,b:max(a,b)/min(a,b);half=args.samples//2
    controls={'AA':ratio(med['A1'],med['A2']),'BB':ratio(med['B1'],med['B2'])};drift={x:ratio(statistics.median(y[:half]),statistics.median(y[half:])) for x,y in values.items()}
    paired=[a/b for suffix in ['1','2'] for a,b in zip(values['A'+suffix],values['B'+suffix])]
    r['timings'].append({'stage':stage,'variant':label,'cold_flush_outside_events':cold,'samples_us':values,'means_us':{x:statistics.mean(y) for x,y in values.items()},'controls':controls,'half_drift':drift,'qualified':max(list(controls.values())+list(drift.values()))<=1.1,'median_paired_speedup_vs_best_triton':statistics.median(paired)});save()
  del graphs,calls,native,w,s,ref,out;gc.collect();torch.cuda.empty_cache()
 r.update(status='COMPLETE_LOCAL_NATIVE_MATERIALIZER_ONLY',strict_bitwise_gate=all(x['bitwise'] and x['finite'] for x in r['checks']),peak_allocated=torch.cuda.max_memory_allocated(),sources_unchanged=all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==v for f,v in r['sources'].items()));save();assert r['sources_unchanged'];print(json.dumps({'status':r['status'],'checks':len(r['checks']),'timings':len(r['timings'])}),flush=True)
if __name__=='__main__':main()

