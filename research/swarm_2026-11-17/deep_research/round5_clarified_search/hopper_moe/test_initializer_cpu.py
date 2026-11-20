"""Public constructor admission exercised with explicit CPU runtime mocks."""
import copy,json,sys,types
from pathlib import Path
import fa3_split_policy as p
N=types.SimpleNamespace;P=Path(__file__).resolve().parent
checks=[]
def make():
    policy=p.TargetFA3SplitPolicy(verify=4,decode=2,enabled=True,collective_setup_agreed=True,precision_binary_audited=True)
    owner=N(_research_fa3_split_policy=policy,is_draft_worker=False)
    b=N(num_splits=0,decode_num_splits=0,is_draft_runner=False,fa_impl_ver=3,use_mla=False,is_encoder_decoder=False,topk=1,kv_cache_is_mxfp8=False,kv_cache_dtype='torch.float8_e4m3fn',head_dim=64,num_attention_heads=16,num_kv_heads=2,decode_cuda_graph_metadata={},target_verify_metadata={},forward_metadata=None,device='cuda:0')
    pa=N(attn_tp_size=8,attn_dp_enabled=False,attn_cp_size=1);ex=N(deterministic=N(enable_deterministic_inference=False),overlap=N(enable_two_batch_overlap=False));di=N(enable_pdmux=False)
    hw=N(ordinal=0,cap=(9,0),name='NVIDIA H200')
    rt=N(get_parallel=lambda:pa,get_exec=lambda:ex,get_disagg=lambda:di)
    torch=N(device=lambda x:N(type=x.split(':')[0],index=int(x.split(':')[1]) if ':' in x else None),cuda=N(current_device=lambda:hw.ordinal,get_device_capability=lambda dev:hw.cap,get_device_name=lambda dev:hw.name))
    return b,owner,pa,ex,di,hw,rt,torch
saved={k:sys.modules.get(k) for k in ('torch','sglang.srt.runtime_context')}
def install(row):sys.modules['sglang.srt.runtime_context']=row[-2];sys.modules['torch']=row[-1]
def reject(name,row):
    install(row)
    try:p.initialize_target_fa3_split_policy(row[0],row[1])
    except p.SplitPolicyError:checks.append(name);return
    raise AssertionError(name+' admitted')
try:
    row=make();install(row);p.initialize_target_fa3_split_policy(*row[:2]);assert p.controller(row[0]);checks.append('public initializer positive with explicitly mocked H200 runtime')
    row[2].attn_cp_size=2
    try:p.controller(row[0]).check()
    except p.SplitPolicyError:checks.append('runtime configuration mutation invalidates controller')
    else:raise AssertionError('CP mutation admitted')
    cases=[('draft worker',1,'is_draft_worker',True),('draft backend',0,'is_draft_runner',True),('FA4',0,'fa_impl_ver',4),('MLA',0,'use_mla',True),('encoder decoder',0,'is_encoder_decoder',True),('cascade',0,'topk',2),('MXFP8',0,'kv_cache_is_mxfp8',True),('BF16 KV',0,'kv_cache_dtype','torch.bfloat16'),('wrong head size',0,'head_dim',128),('wrong head geometry',0,'num_attention_heads',8),('existing graph metadata',0,'decode_cuda_graph_metadata',{1:object()}),('existing metadata',0,'forward_metadata',object()),('wrong device ordinal',0,'device','cuda:1'),('SM86',5,'cap',(8,6)),('H100',5,'name','NVIDIA H100'),('TP4',2,'attn_tp_size',4),('DP',2,'attn_dp_enabled',True),('CP',2,'attn_cp_size',2),('PDMux',4,'enable_pdmux',True)]
    for name,i,key,value in cases:
        row=make();setattr(row[i],key,value);reject(name,row)
    for group,key in [('deterministic','enable_deterministic_inference'),('overlap','enable_two_batch_overlap')]:
        row=make();setattr(getattr(row[3],group),key,True);reject(key,row)
    for flag in ('collective_setup_agreed','precision_binary_audited'):
        row=make();data=dict(verify=4,decode=2,enabled=True,collective_setup_agreed=True,precision_binary_audited=True);data[flag]=False;row[1]._research_fa3_split_policy=p.TargetFA3SplitPolicy(**data);reject(flag,row)
    row=make();row[1]._research_fa3_split_policy=p.TargetFA3SplitPolicy();install(row);p.initialize_target_fa3_split_policy(*row[:2]);assert p.controller(row[0]) is None;checks.append('disabled public initializer leaves backend untouched')
finally:
    for k,v in saved.items():
        if v is None:sys.modules.pop(k,None)
        else:sys.modules[k]=v
report={'status':'CPU_PUBLIC_INITIALIZER_MOCK_CONTRACT_PASS','count':len(checks),'checks':checks,'scope':'Real initializer and snapshot check with mocked torch/runtime context; no native backend constructor, CUDA device or all-rank agreement qualification.'}
(P/'CPU_INITIALIZER_TESTS.json').write_text(json.dumps(report,indent=2));print(report['status'],len(checks))
