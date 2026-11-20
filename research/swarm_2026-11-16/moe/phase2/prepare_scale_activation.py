"""Build isolated exact-scale experiment; never changes scored baseline."""
import ast,hashlib,json,re
from pathlib import Path
P=Path(__file__).resolve().parent
S=P.parents[1]/'coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang'
split=(S/'kernels/ops/moe/muse_spark_humming_split.py').read_text()
core=(S/'srt/layers/muse_spark/muse_spark_nvfp4_moe.py').read_text()
node=next(n for n in ast.parse(core).body if isinstance(n,ast.FunctionDef) and n.name=='_humming_expert_outputs_default')
default='\n'.join(core.splitlines()[node.lineno-1:node.end_lineno])+'\n'
pattern=r'(?m)^    scale_active_rows\(cache1,\s*local_ids,\s*layer.w13_post_scale\)\n(    cache2 = [^\n]+)\n    silu_and_mul\(cache1,\s*cache2,\s*expert_ids=local_ids.flatten\(\),\s*expert_step=1\)'
def fuse(source):
    result,count=re.subn(pattern,r'\1\n    scaled_silu_and_mul(cache1,cache2,local_ids,layer.w13_post_scale)',source)
    assert count==1,count
    return result
candidate=fuse(split)+'\n\n'+fuse(default)+'\n\n'+'''def fused_expert_outputs(layer,x,local_ids,torch,global_experts=None,*,retain=False,defer_downscale=False):
    return split_expert_outputs(_humming_expert_outputs_default,layer,x,local_ids,torch,
        global_experts,retain=retain,defer_downscale=defer_downscale)
'''
candidate=candidate.replace('import torch\n','import torch\nfrom scale_activation import scaled_silu_and_mul\n',1)
(P/'scale_activation_helper.py').write_text(candidate)
ast.parse(candidate)
print('Built and syntax-checked helper only; fullchain harness is maintained separately')
