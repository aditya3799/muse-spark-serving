"""Primary-source dispatch, precision and baseline-inclusion audit; CPU only."""
import ast,hashlib,json,subprocess,urllib.request
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1];out=here/'pr51393'
pin='36c4cd4d2e8fa600bd5d1587c9b6fc5fd903fe59';base='297e54fd105e974cdf401e5b4d20253a22c16f90'
files=['python/sglang/kernels/ops/moe/nvfp4_w4a16_sm90.py','python/sglang/kernels/jit/csrc/moe/nvfp4_w4a16_sm90.cuh']
snap=[]
for name in files:
 url='https://raw.githubusercontent.com/sgl-project/sglang/'+pin+'/'+name
 with urllib.request.urlopen(url,timeout=40) as r:data=r.read(600001)
 assert len(data)<=600000
 path=out/Path(name).name;path.write_bytes(data)
 present=subprocess.check_output(['git','ls-tree',base,name],cwd=root/'sglang')
 snap.append(dict(path=name,url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),present_in_frozen_tree=bool(present)))
op=(out/'nvfp4_w4a16_sm90.py').read_text();tree=ast.parse(op)
names={'DECODE_MAX_TOKENS','_DF_FIXED','_DF_RING','_DF_SMEM'};keep=[]
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in ('_decode_fused_fits','_decode_variant'):keep.append(node)
 if isinstance(node,ast.Assign):
  targets={n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name)}
  if targets&names:keep.append(node)
env={};exec(compile(ast.Module(body=keep,type_ignores=[]),'actual_upstream_dispatch','exec'),env)
cases=[]
for m in (1,2,3,4,8,32,128,256,753):
 choice=env['_decode_variant'](m,m*8,32,4096,4096)
 assert choice==('decode_fused' if m<=2 else 'auto')
 cases.append(dict(tokens=m,topk=8,local_experts=32,H=4096,I=4096,choice=choice))
old_model=(here/'muse_logits_activation/muse_spark.py').read_text()
sp=subprocess.check_output(['git','show',base+':python/sglang/srt/layers/layernorm_sp.py'],cwd=root/'sglang').decode()
assert 'forward_mode == ForwardMode.EXTEND' in sp
(out/'pinned_layernorm_sp.py').write_text(sp)
code=(out/'nvfp4_w4a16_sm90.cuh').read_text()
assert 'DF_TMAX = 4' in code and 'DP_TMAX = 4' in code
parent=root/'release_2026-11-18/candidate_sources/python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
receipt=dict(head=pin,frozen_base=base,files=snap,dispatch_cases=cases,cxx_hardmax_tokens=4,
 baseline_tree_has_backend=False,graph_ancestry_not_locally_available=True,
 layernorm_sp_prefill_only=True,verify_token_formula='batch * unchanged block_size (8 in this research protocol)',
 actual_dflash_min_verify_tokens=8,eligible_auto_decode_fastpath=False,
 precision_difference='Upstream FP32 gate/up accumulation times FP32 global scale feeds SiLU before BF16 cast; frozen Humming has BF16 GEMM output then separate BF16 scale before SiLU. Upstream fused weighted sum also bypasses Muse per-expert post-norm.',
 parent_moe_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
 local_gpu_or_hardware_execution=False,serving_gain_measured=False)
(out/'audit.json').write_text(json.dumps(receipt,indent=2));print('PASS actual donor dispatch9 shapes; minimum verifyM8 excluded, maxM4, frozen backend files absent')
