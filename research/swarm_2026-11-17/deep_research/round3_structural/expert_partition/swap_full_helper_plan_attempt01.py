"""Frozen SM86 scored E8/M32 full-helper policy, without CUDA imports."""
from pathlib import Path
import copy,hashlib,importlib.util,json
P=Path(__file__).resolve().parent;ROOT=P.parents[3];HOPPER=P.parent/'hopper_moe'
CONFIG=ROOT/'swarm_2026-11-17/deep_research/expert_partition/k_preserve_m1024_01/owned_configs.json'
BASE=HOPPER/'SM86_COMPACT_E8_W13_JOBS.json'

def plan(variant='swap_tiles'):
 assert variant in ('reference','swap_same','swap_tiles')
 intervals=json.loads(CONFIG.read_text())['indexed'];base=json.loads(BASE.read_text())['jobs'][0]['configuration']
 result={}
 for name,tune in (('light',8),('heavy',256)):
  result[name]={}
  for stage,n in (('w13',8192),('w2',4096)):
   c=copy.deepcopy(next(c for lo,hi,c in intervals[stage] if lo<tune<=hi))
   original=copy.deepcopy(c)
   if variant=='swap_tiles':
    bm=8 if name=='light' else24
    c['block_shape'][0]=bm;c['warp_shape'][0]=bm
   assert c['block_shape'][2]==original['block_shape'][2] and c['warp_shape'][2]==original['warp_shape'][2]
   full=copy.deepcopy(base);full.update(c);full.pop('num_sms',None);full.update(shape_n=n,shape_k=4096,num_experts=8)
   result[name][stage]=full
 return {'light':result['light'],'heavy':result['heavy'],'cutoff':16}

def admission():
 spec=importlib.util.spec_from_file_location('fullhelper_review_admit',HOPPER/'swap_ab_humming_codegen.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 checks=[]
 for arm in ('swap_same','swap_tiles'):
  for partition,stages in plan(arm).items():
   if partition=='cutoff':continue
   for stage,c in stages.items():
    try:mod.admit(c);state='ADMITTED';error=None
    except ValueError as exc:state='HELD_RENDERER_REJECTED';error=str(exc)
    checks.append(dict(arm=arm,partition=partition,stage=stage,status=state,error=error,block_shape=c['block_shape'],warp_shape=c['warp_shape'],N=c['shape_n'],K_banks=c['block_shape'][2]//c['warp_shape'][2]))
 return checks

def manifest():
 checks=admission();return {'status':'CPU_PREPARED_RUNTIME_HELD' if any(c['status']!='ADMITTED'for c in checks) else 'RENDERER_ADMITS_REQUIRES_NUMERICAL_GATES_AND_ROOT_LEASE','scope':'FrozenE32 SM86 tables, actualscored E8/M32 top8/global256 helper semantics; expectedlocal tune_m8 anddefault256; lighttune8equals32 inbothstages. ReducedE8fixture notnativeE32/H200/serving.','reference':plan('reference'),'swap_same':plan('swap_same'),'swap_tiles':plan('swap_tiles'),'admission':checks,'sources':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [CONFIG,BASE,HOPPER/'swap_ab_humming_codegen.py',Path(__file__)]},'required_gates':['WK32 two32wideKbanks forlight; changingtoWK64wouldchangebaselineK arithmetic','W2N4096 rendereradmission andnativeactivation-fed input exactness','Combined fullhelper strictbitwise stage/final checks beforecleanpairedtiming','RootexclusiveGPU/compilerlease and512MiBfree reserve'], 'memory':{'production':'Onecanonical packedcopy W13/W2 andexistingtwohelperworkspaces; no extraresident fullweights','validation_weights_bytes':408*1024**2,'weight_breakdown':'192MiBreferencepacked+192MiBcanonicalpacked+24MiBsharedFP8scales forE8H=I4096','graph_scratch':'EachM32 top8 originalhelper3*S*H*2=6MiB plusfinal/outputobserver; numericgraphsdeletedbeforetiming','target_peak_bytes':700*1024**2,'hard_own_budget_bytes':int(1.15*1024**3),'free_reserve_bytes':512*1024**2,'construction':'Chunkcanonicalwordtranspose <=1Mwords; delete rawfixture beforecanonicalcopy. Avoid fullweight int64 conversion.'},'arms':{'A':'actualscored split helperreference,originalpackedweights','B':'swap orientation only,originalBM16 light/heavy,Ktopologyunchanged','C':'swapBM8light/BM24heavy,originalcutoff16 andKtopologyunchanged'},'routes':['legalglobaltop8withoutreplacementuniform256','all8distinctlocalhot','mixedcutoff16/17 andlast-expert','allremote','tailinvalid'],'timing':'Cleancompletehelper graphs inclhistogram, eacharmownalignments, bothW13,globalBF16scale,nativeSiLU,bothW2,scoredfinalizer. Uniform/hot/mixedseparate. InterleavedA1/A2/B1/B2/C1/C2 controls, pairedrawsamples, warm/coldflushoutsideevents, firsthalf/lasthalfdrift; no productofcomponentratios.'}
if __name__=='__main__':
 r=manifest();(P/'SWAP_FULL_HELPER_PLAN.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'admission':r['admission']}))
