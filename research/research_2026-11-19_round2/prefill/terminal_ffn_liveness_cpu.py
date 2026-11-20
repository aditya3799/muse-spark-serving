"""Actual logits-consumer AST liveness proof using row-identity CPU surrogates."""
import ast,json,hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SRC=ROOT/'sglang/python/sglang/srt/layers/logits_processor.py'
CONFIG=HERE/'sources/nov19_refresh/muse_dflash_config_46c90f.json'
class Rows:
 def __init__(self,rows,width=8192):self.rows=list(rows);self.shape=(len(self.rows),width)
 def __getitem__(self,k):
  if isinstance(k,Rows):k=k.rows
  if isinstance(k,list):return Rows([self.rows[i] for i in k],self.shape[1])
  return Rows(self.rows[k] if isinstance(k,slice) else [self.rows[k]],self.shape[1])
 def __sub__(self,x):return Rows([v-x for v in self.rows],self.shape[1])
class Mode:
 def __init__(self,name):self.name=name
 def is_decode_or_idle(self):return self.name in ('decode','idle')
 def is_target_verify(self):return self.name=='verify'
 def is_draft_extend_v2(self):return False
 def is_extend(self):return self.name=='extend'
class Capture:
 def need_capture(self):return True
 def is_full(self):return True
 def is_last(self):return False
def cumsum(x,dim=0):
 result=[];total=0
 for v in x.rows:total+=v;result.append(total)
 return Rows(result)
ns=dict(torch=NS(Tensor=Rows,cumsum=cumsum),Optional=Optional,AuxHiddenStates=object,LogitsMetadata=object,pack_aux_hidden_states=lambda x:x)
tree=ast.parse(SRC.read_text()); names=['_get_pruned_states','_get_hidden_states_to_store','_get_last_hidden_states_to_store']
for name in names:
 node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
 mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
 exec(compile(ast.fix_missing_locations(mod),str(SRC),'exec'),ns)
config=json.loads(CONFIG.read_text());ids=config['dflash_config']['target_layer_ids'];last=config['num_target_layers']-1
assert ids==[1,16,30,44,59] and last==61 and last not in ids
results=[]
for lens in ([753],[751,754],[1,8,32,128]):
 total=sum(lens);hidden=Rows(range(total));aux=[Rows([f'{layer}:{r}' for r in range(total)]) for layer in ids]
 for mode in ('extend','verify'):
  md=NS(forward_mode=Mode(mode),draft_extend_select_index=None,extend_return_logprob=False,extend_seq_lens=Rows(lens),capture_hidden_mode=Capture())
  pruned,before,aux_pruned,sample,*_=ns['_get_pruned_states'](NS(),hidden,None,aux,md)
  if mode=='verify':expected=list(range(total))
  else:
   expected=[];offset=0
   for n in lens:offset+=n;expected.append(offset-1)
  assert pruned.rows==expected
  stored=ns['_get_hidden_states_to_store'](NS(),hidden,None,aux,pruned,before,aux_pruned,sample,md)
  assert stored is aux and all(len(x.rows)==total for x in stored)
  assert ns['_get_last_hidden_states_to_store'](NS(keep_last_hidden_states=False),hidden,aux,md) is None
  assert ns['_get_last_hidden_states_to_store'](NS(keep_last_hidden_states=True),hidden,aux,md) is hidden
  results.append(dict(mode=mode,lens=list(lens),final_rows_for_logits=expected,aux_full_rows=total,final_hidden_only_capture_feature=True,passed=True))
doc=dict(passed=True,scope=__doc__,config_target_layers=ids,terminal_layer=last,cases=results,
 sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SRC,CONFIG)},
 conclusions=['DFLASH FULL capture returns all auxiliary rows, not automatically all final-layer rows.','Ordinary prefill logits consume the final new row per request.','Target verify consumes every target row: all8/block remain live.','Capture-only keep_last_hidden_states activates a separate all-final-row consumer and must veto pruning.'],
 limitations=['Source consumer proof with row-identity surrogates; not a TP/BCG model implementation or numerical test.','Final layer timing and checkpoint endpoint dtype remain unknown.','Any new full-hidden/logprob/embedding consumer invalidates this narrow liveness predicate.'])
(HERE/'terminal_ffn_liveness_cpu.json').write_text(json.dumps(doc,indent=2));print(json.dumps(dict(passed=True,cases=len(results),target_layers=ids,terminal=last)))
