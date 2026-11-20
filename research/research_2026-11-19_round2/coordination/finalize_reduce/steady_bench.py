import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import probe as p
import torch,ctypes as ct,json,time,math,statistics,numpy as np,gc,resource,hashlib
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
HERE=p.HERE;native=p.native;Args=p.Args;run_cuda=p.run_cuda

def ci(values):
 a=np.log(np.array(values));rng=np.random.default_rng(239917);samples=np.exp(a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1));return [float(x) for x in np.quantile(samples,[.025,.975])]
def gm(v):return math.exp(statistics.mean(math.log(x) for x in v))
@torch.no_grad()
def main():
 torch.set_num_threads(2);h=4096;rank=3
 report={'scope':p.__doc__ or 'Virtual-peer local chain only; no transport','reason':'First timing invalid because duplicateparent controls0.85..1.49; unchanged kernel, longer warmed paired timing','source_sha256':hashlib.sha256((HERE/'combine_reduce.cu').read_bytes()).hexdigest(),'criteria':{'control_gm_range':[.97,1.03],'control_CI_range':[.95,1.05],'aggregate_gain_CI_low_min':1.01,'case_gain_CI_low_min':.98},'cases':[],'complete':False}
 path=HERE/'steady_bench.json'
 def save():path.write_text(json.dumps(report,indent=2)+'\n')
 save()
 for m in (2,3,8):
  for scale in (False,True):
   torch.manual_seed(819103+m+scale)
   x=torch.empty((m,8,h),device='cuda',dtype=torch.bfloat16);weight=torch.rand(h,device='cuda',dtype=torch.bfloat16)+.5;scales=torch.rand(32,device='cuda',dtype=torch.bfloat16)+.5
   scores=torch.empty((m,8),device='cuda');ids=torch.empty((m,8),device='cuda',dtype=torch.int32);peers=torch.empty((8,m,h),device='cuda',dtype=torch.bfloat16);inv=torch.empty((m,8),device='cuda');local=torch.empty((m,h),device='cuda',dtype=torch.bfloat16)
   outputs={n:torch.empty_like(local) for n in ('native','candidate','control')};params={n:Args(x.data_ptr(),weight.data_ptr(),scales.data_ptr(),peers.data_ptr(),local.data_ptr(),scores.data_ptr(),inv.data_ptr(),ids.data_ptr(),o.data_ptr(),None,m,h,rank) for n,o in outputs.items()}
   def run(n):
    native._post_expert_inv_rms_kernel[(m*8,)](x,weight,ids,inv,scales,HAS_SCALE=scale,TOPK=8,H=h,eps=1e-8,HAS_WEIGHT=True,WEIGHT_FIRST=True,BLOCK_H=h,num_warps=8,num_stages=2)
    if n!='candidate':native._post_expert_tiled_combine_kernel[(m,h//256)](x,weight,scores,ids,inv,local,scales,HAS_SCALE=scale,TOPK=8,H=h,HAS_WEIGHT=True,WEIGHT_FIRST=True,BLOCK=256,num_warps=4)
    run_cuda(n=='candidate',True,True,scale,params[n])
   def update(i):
    torch.manual_seed(819103+m*41+i*8191+scale);x.normal_();peers.normal_();scores.uniform_(0,.3);z=torch.randint(0,256,(m,8),device='cuda',dtype=torch.int32)-rank*32;ids.copy_(torch.where((z>=0)&(z<32),z,-1));x[ids<0]=float('nan')
   update(0);graphs={}
   for n in params:
    run(n);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
     for _ in range(128):run(n)
    graphs[n]=g
   # A fixed warmup; no clocks/power policy changed.
   for _ in range(20):
    for g in graphs.values():g.replay()
   torch.cuda.synchronize();rounds=[];all_exact=True
   for i in range(12):
    update(i+1);times={n:[] for n in params}
    order=('native','candidate','control','control','candidate','native') if i%2==0 else ('control','candidate','native','native','candidate','control')
    for n in order:
     b=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);b.record()
     for _ in range(16):graphs[n].replay()
     e.record();e.synchronize();times[n].append(b.elapsed_time(e)*1000/(16*128))
    exact=torch.equal(outputs['native'].view(torch.int16),outputs['candidate'].view(torch.int16)) and torch.equal(outputs['native'].view(torch.int16),outputs['control'].view(torch.int16));all_exact &=exact
    mean={n:statistics.mean(v) for n,v in times.items()};rounds.append({'times_us':mean,'gain':mean['native']/mean['candidate'],'control':mean['native']/mean['control'],'exact':exact})
   gains=[v['gain'] for v in rounds];controls=[v['control'] for v in rounds];row={'m':m,'h':h,'scale':scale,'rounds':rounds,'all_exact':all_exact,'gain':gm(gains),'gain_ci':ci(gains),'control':gm(controls),'control_ci':ci(controls)}
   row['valid_control']=.97<=row['control']<=1.03 and row['control_ci'][0]>=.95 and row['control_ci'][1]<=1.05
   report['cases'].append(row);save();print(m,scale,json.dumps({k:row[k] for k in ('gain','gain_ci','control','control_ci','valid_control','all_exact')}),flush=True)
   del graphs,g,params,outputs,x,weight,scales,scores,ids,peers,inv,local;gc.collect();torch.cuda.empty_cache()
 report['complete']=True;report['all_exact']=all(c['all_exact'] for c in report['cases']);report['valid_controls']=all(c['valid_control'] for c in report['cases']);report['aggregate_gain']=gm([c['gain'] for c in report['cases']]);report['aggregate_ci']=ci([v['gain'] for c in report['cases'] for v in c['rounds']]);report['local_keep']=report['all_exact'] and report['valid_controls'] and report['aggregate_ci'][0]>1.01 and all(c['gain_ci'][0]>.98 for c in report['cases']);report['serving_qualified']=False;save()
if __name__=='__main__':main()
