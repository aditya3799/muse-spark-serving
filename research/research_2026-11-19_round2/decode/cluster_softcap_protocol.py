"""CPU protocol/value proof only; does not execute CUDA barriers or softcap math."""
import hashlib,itertools,json,math,random,struct
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'cluster_softcap';out.mkdir(exist_ok=True)
def bits(value):return struct.pack('<f',value).hex()
def native_fold(partials,offset=0):
 best,index=-math.inf,0
 for value,idx in partials:
  if value>best:best,index=value,idx
 return bits(best),index+offset
def protocol(partials,order,offset=0,final_barrier=True):
 storage={};alive=set(range(8));published=set();steps=[]
 for rank in order:
  storage[rank]=partials[rank];published.add(rank)
 assert len(published)==8 # first barrier must complete before remote reads
 if not final_barrier:
  # A legal scheduling interleaving without the final lifetime barrier.
  victim=next(rank for rank in order if rank!=0);alive.remove(victim)
 result=[]
 for rank in range(8):
  if rank not in alive:raise RuntimeError('remote shared memory owner exited before its read')
  result.append(storage[rank])
 answer=native_fold(result,offset)
 alive.clear() # second barrier releases all CTAs only after every read
 return answer
rng=random.Random(623791);cases=[]
for count in range(9):
 for pattern in ('finite','nan_all','nan_first','nan_last','tie','signed_zero','negative_inf','positive_inf'):
  values=[rng.uniform(-20,20) for _ in range(count)]
  if pattern=='nan_all':values=[math.nan]*count
  if pattern=='nan_first' and count:values[0]=math.nan
  if pattern=='nan_last' and count:values[-1]=math.nan
  if pattern=='tie':values=[1.0]*count
  if pattern=='signed_zero':values=[-0.0 if i%2==0 else 0.0 for i in range(count)]
  if pattern=='negative_inf':values=[-math.inf]*count
  if pattern=='positive_inf':values=[math.inf]*count
  partials=[(v,4096*i+17) for i,v in enumerate(values)]+[(-math.inf,0)]*(8-count)
  order=list(range(8));rng.shuffle(order)
  assert protocol(partials,order,offset=25256)==native_fold(partials[:count],offset=25256)
  cases.append(dict(valid_tiles=count,pattern=pattern,arrival=order,exact=True))
# Every arrival permutation; tile values deliberately combine NaNs, ties and zeros.
partials=[(math.nan,0),(-0.0,4096),(0.0,8192),(9.0,12288),(9.0,16384),(-math.inf,20480),(math.nan,24576),(-math.inf,0)]
expected=native_fold(partials);schedules=0
for order in itertools.permutations(range(8)):
 assert protocol(partials,order)==expected;schedules+=1
try:protocol(partials,list(range(8)),final_barrier=False)
except RuntimeError as error:missing_final_barrier=str(error)
else:raise AssertionError('missing lifetime barrier was not detected')
outcome=dict(scope=__doc__,passed=True,cases=cases,all_arrival_permutations=schedules,
 missing_final_barrier_counterexample=missing_final_barrier,
 inactive_tile_identity=['-inf',0],fold='sequential strict > in original tile order, initial(-inf,0)',
 original_pointwise_math_tested=False,original_block_argmax_tested=False,
 cuda_cluster_execution=False,hardware_timing=False,
 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 donor_sha256=hashlib.sha256((here/'flashinfer_nov/sampling.cuh').read_bytes()).hexdigest())
(out/'cpu_protocol.json').write_text(json.dumps(outcome,indent=2));print('PASS',len(cases),'value cases,',schedules,'arrival schedules; missing final barrier rejected')
