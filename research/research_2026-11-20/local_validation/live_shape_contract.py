"""CPU negative cases: graph padding is trimmed before exact-shape adapter lookup."""
import json,types
import torch
import adapter_probe as ap
from exported_chain import method
R=ap.R;O=R/'research_2026-11-20/local_validation'
project=method(ap.SOURCE/'srt/models/dflash.py','project_target_hidden')
append=method(ap.SOURCE/'srt/speculative/dflash_worker_v2.py','_append_target_hidden_to_draft_kv_by_loc')
class Linear(torch.nn.Linear):
    def __init__(self):super().__init__(8,4,bias=False);self.calls=[]
    def forward(self,x):self.calls.append(x.shape[0]);return super().forward(x)
class Group:
    world_size=8;rank_in_group=0;cpu_group=None
    def all_gather(self,*a,**k):raise AssertionError('unregistered shape entered a collective')
records=[]
for prefill,padded,live in ((True,768,753),(True,768,752),(True,3072,3012),(False,32,24)):
    linear=Linear();group=Group()
    row=ap.row.RowProjectionRouter(linear,group,empirical=True);row.published={(768,8),(3072,8),(16384,8)}
    col=ap.nmod.OutputNRouter(linear,group,mode='qualified');col.entries={(8,8):None,(32,8):None}
    model=types.SimpleNamespace(fc=linear,quantizable=False,hidden_norm=lambda z:z,num_context_features=2,config=types.SimpleNamespace(hidden_size=4))
    model.project_target_hidden=types.MethodType(project,model)
    outputs=[]
    worker=types.SimpleNamespace(draft_model=model,model_runner=types.SimpleNamespace(device=torch.device('cpu')),draft_parallel_layout=None,lilicorr=None,_use_fused_kv_materialize=False,_fused_kv_helper=None,_prefill_row_projection=row,_decode_n_projection=col,_append_target_hidden_sequential=lambda **k:outputs.append(k['ctx_hidden']))
    x=torch.randn((padded,8));ids=torch.arange(live)
    append(worker,target_hidden=x,cache_loc=ids,positions=ids,prefill_projection=prefill)
    assert linear.calls==[live]
    assert torch.equal(outputs[0],torch.nn.functional.linear(x[:live],linear.weight))
    records.append(dict(prefill=prefill,padded_rows=padded,live_rows=live,native_fallback=True,correct=True))
doc=dict(complete=True,correct=True,scope=__doc__,examples_only=True,serving_dispatch_frequency_measured=False,cases=records)
(O/'live_shape_contract.json').write_text(json.dumps(doc,indent=2));print(json.dumps(doc))
