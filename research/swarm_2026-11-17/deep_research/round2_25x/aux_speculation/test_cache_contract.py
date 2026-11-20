import ast,__future__,json
from pathlib import Path
from types import SimpleNamespace as NS
P=Path(__file__).resolve().parent;S=P.parents[3]/'sglang/python/sglang/srt'
def extract(path,name,ns,cls=None):
 body=ast.parse(path.read_text(encoding='utf-8')).body
 if cls:body=next(n.body for n in body if isinstance(n,ast.ClassDef) and n.name==cls)
 node=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name);node.decorator_list=[]
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec',flags=__future__.annotations.compiler_flag),ns);return ns[name]
schedule=NS(max_running_requests=32,chunked_prefill_size=16384);memory=NS(disable_radix_cache=False);disagg=NS(disaggregation_mode='null')
kvc=NS(sliding_window_size=2048,layer_info=NS(full_attention_layer_ids=[1]))
app=extract(S/'model_executor/pool_configurator.py','is_applicable',{'get_schedule':lambda:schedule,'get_memory':lambda:memory,'get_disagg':lambda:disagg},'SWARequestCapPoolConfigurator')
checks=[]
def check(name,ok):checks.append({'name':name,'passed':bool(ok)});assert ok,name
check('scored_radix_rejects_request_cap',not app(kvc))
memory.disable_radix_cache=True;check('radix_disabled_chunked_cap_control',app(kvc))
schedule.chunked_prefill_size=0;check('unchunked_prefill_rejected',not app(kvc))
disagg.disaggregation_mode='decode';check('decode_mode_unchunked_control',app(kvc))
schedule.max_running_requests=None;check('explicit_request_limit_required',not app(kvc));schedule.max_running_requests=32
kvc.sliding_window_size=None;check('window_required',not app(kvc));kvc.sliding_window_size=2048
kvc.layer_info.full_attention_layer_ids=[];check('full_attention_component_required',not app(kvc))
CT=NS(SWA='swa')
free=extract(S/'mem_cache/common.py','free_swa_out_of_window_slots',{'ComponentType':CT,'ceil_align':lambda n,p:((n+p-1)//p)*p})
class KV:
 holds_kv=True;cache_protected_len=2048;req_pool_idx=7
 def __init__(self):self.evicted=0
 def get_evicted_seqlen(self,c):return self.evicted
 def set_evicted_seqlen(self,c,n):self.evicted=n
 def swa_dead_lo(self,page):return self.cache_protected_len
class Rows:
 def __getitem__(self,key):return list(range(key[1].start,key[1].stop))
freed=[];req=NS(kv=KV());pool=NS(req_to_token=Rows());alloc=NS(free_swa_segment=lambda slots,start_pos:freed.append((start_pos,len(slots))))
kw=dict(sliding_window_size=2048,page_size=128,req_to_token_pool=pool,token_to_kv_pool_allocator=alloc)
free(req,8191,**kw);check('never_free_radix_protected_prefix',freed==[(2048,3968)]);check('whole_window_page_floor',req.kv.evicted==6016)
free(req,8191,**kw);check('same_frontier_not_double_freed',len(freed)==1)
free(req,8319,**kw);check('advancing_frontier_only_new_page',freed[-1]==(6016,128))
req=NS(kv=KV());freed.clear();free(req,100000,retain_floor=4096,**kw);check('checkpoint_retain_floor',freed==[(2048,2048)])
req.kv.holds_kv=False;free(req,200000,**kw);check('no_kv_lease_no_free',len(freed)==1)
req=NS(kv=KV());req.kv.cache_protected_len=1;rejected=False
try:free(req,8191,**kw)
except AssertionError:rejected=True
check('unaligned_prefix_rejected',rejected)
report={'status':'PASS','checks':checks,'check_count':len(checks),'scope':'Extracted actual applicability and free-frontier methods; Python range allocator fixture. No CUDA attention, HiCache transfers, allocator physical mappings or scheduler overlap test.'}
(P/'CACHE_CPU_CONTRACT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','checks':len(checks)}))
