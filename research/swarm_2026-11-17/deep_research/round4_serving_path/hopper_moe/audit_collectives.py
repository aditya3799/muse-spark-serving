import ast, hashlib, json, pathlib, types
ROOT=pathlib.Path(r'D:\OpenKernelChallenge\serving')
HERE=pathlib.Path(__file__).resolve().parent
paths=[
'sglang/python/sglang/srt/models/muse_spark.py',
'sglang/python/sglang/srt/models/dflash.py',
'sglang/python/sglang/srt/speculative/draft_parallel.py',
'sglang/python/sglang/srt/distributed/parallel_state.py',
'sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py',
'sglang/python/sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py',
'sglang/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh',
'sglang/python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh',
'sglang/python/sglang/kernels/jit/include/sgl_kernel/vec.cuh',
'sglang/python/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh',
'sglang/python/sglang/kernels/ops/layernorm/muse_spark_norm_residual_jit.py',
'sglang/python/sglang/kernels/ops/communication/muse_spark_all_gather.py',
'sglang/python/sglang/kernels/ops/communication/muse_spark_all_gather_norm_residual.py',
]
# Execute only source definitions with no torch import or CUDA queries.
p=ROOT/paths[5];tree=ast.parse(p.read_text(encoding='utf-8'))
keep=[]
for n in tree.body:
    if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and getattr(n,'name','') not in ['_get_all_reduce_configs','get_supported_world_sizes','get_all_reduce_config']:
        keep.append(n)
    elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) for t in n.targets):keep.append(n)
from typing import NamedTuple,Optional
ns=dict(NamedTuple=NamedTuple,Optional=Optional,cache=lambda f:f)
exec(compile(ast.Module(body=keep,type_ignores=[]),str(p),'exec',flags=__import__('__future__').annotations.compiler_flag),ns)
raw_cfg=ns['_sm90_configs'](132)[8]
cfg=raw_cfg.clip(max_push_bytes=min(raw_cfg.max_push_bytes(8),16*1024*1024),max_pull_bytes=min(raw_cfg.max_pull_bytes,16*1024*1024),world_size=8)
def select(b,h,mc):
    if b<=h.one_shot_push_threshold:return 'ONE_SHOT_PUSH'
    if b<=h.two_shot_push_threshold:return 'TWO_SHOT_PUSH'
    if b<=h.one_shot_pull_threshold:return 'ONE_SHOT_PULL'
    if mc and h.mc.contains(b):return 'TWO_SHOT_PULL_MULTICAST'
    if b<=h.two_shot_pull_threshold:return 'TWO_SHOT_PULL_UNICAST'
    return 'NCCL_FALLBACK'
rows=[]
for m in [1,8,9,16,32,64,128,256,512,753,3012,8192,16384]:
 for h in [4096,8192]:
  for graph in [False,True]:
   for mc in [False,True]:
    rows.append(dict(M=m,H=h,bytes=m*h*2,graph=graph,multicast=mc,algorithm=select(m*h*2,cfg.graph if graph else cfg.eager,mc)))
report=dict(status='SOURCE_ONLY_NO_LIVE_MULTIGPU_TELEMETRY',source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},AGENTS='none found in examined ancestor/project paths',raw_sm90_tp8_config=raw_cfg._asdict(),sm90_tp8_config=cfg._asdict(),capacity_assumption='default16MiB max_size and enabledV2; explicit overrides not observed',dispatch_rows=rows,shortlist=[
 dict(candidate='attention AR plus original Muse JIT boundary',scope='TP8 BF16 H8192 M1..8 one-shot-push; no SP/DP; preserve norm body and rank-order reduce_vec',missing='GPU peer mapping/replay and H200 timing; current runtime communicator/backend telemetry'),
 dict(candidate='pull or multicast AR plus original norm consumer',scope='broader M; must preserve stock multicast reduction primitive and two-shot protocol',missing='same arithmetic/epoch proof and storage reuse safety'),
 dict(candidate='draft AR plus residual/RMS',scope='5 draft layers, attention and MLP each reduce',missing='draft model-specific conv guards and norm exact operation order'),
],rejected=[dict(candidate='move post-expert projection before AR',reason='BF16 all-reduce result is rounded before projection; distributivity is not finite-precision identity'),dict(candidate='normalize rank-local partial before AR',reason='RMS normalization is nonlinear'),dict(candidate='remove draft TP collectives because SINGLE_GROUP',reason='without DP/CP collapse SINGLE_GROUP retains target attention TP group')],constraints=dict(active_source_changes=False,remote_calls=False,submissions=False,GPU=False,compiler=False))
(HERE/'SOURCE_COLLECTIVE_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(rows=len(rows),push_config=cfg.graph.one_shot_push_threshold,source_files=len(paths))))


