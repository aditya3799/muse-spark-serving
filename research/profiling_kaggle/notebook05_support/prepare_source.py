"""Prepare a V4-relative draft and executable source slices; never edit the checkout."""
import ast
import difflib
import hashlib
import json
import textwrap
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving')
REL='python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'
baseline=(ROOT/'sglang'/REL).read_text(encoding='utf-8')
previous=json.loads((ROOT/'profiling_kaggle/notebook04_support/source_manifest_04.json').read_text())
assert hashlib.sha256((ROOT/'sglang'/REL).read_bytes()).hexdigest()==previous['owner_source_sha256']
marker='        self.register_buffer("_expert_map", expert_map, persistent=False)\n'
assert baseline.count(marker)==1
addition='''        # Derived from the fixed expert map; not part of model checkpoints.
        expert_lookup = (
            None
            if expert_map is None
            else torch.cat((expert_map.new_full((1,), -1), expert_map))
        )
        self.register_buffer("_expert_lookup", expert_lookup, persistent=False)
'''
methods='''    @torch.no_grad()
    def refresh_expert_lookup_(self) -> None:
        """Refresh after map edits, before consumers on the ordered CUDA stream.

        Preserves captured storage. Replacing/moving buffers requires graph
        recapture; direct map mutations are not automatically detected.
        """
        if self._expert_map is not None:
            self._expert_lookup[0].fill_(-1)
            self._expert_lookup[1:].copy_(self._expert_map)

    @torch.no_grad()
    def update_expert_map_(self, mapping: torch.Tensor) -> None:
        """Update fixed-shape routing metadata without replacing graph inputs."""
        if self._expert_map is None:
            raise ValueError("Mapping cannot be enabled without reconstructing the module")
        if (
            mapping.shape != self._expert_map.shape
            or mapping.dtype != torch.int32
            or mapping.device != self._expert_map.device
        ):
            raise ValueError("Expert map update must preserve shape, int32 dtype and device")
        self._expert_map.copy_(mapping)
        self.refresh_expert_lookup_()

'''
old='''                local_topk_ids = _map_topk_ids_to_local_experts(
                    local_topk_ids, expert_map
                )'''
new='''                local_topk_ids = self._expert_lookup[
                    local_topk_ids.clamp_min(-1) + 1
                ]'''
assert baseline.count(old)==1
candidate=baseline.replace(marker,marker+addition).replace('    def _marlin_expert_outputs(',methods+'    def _marlin_expert_outputs(',1).replace(old,new)
unused='''        from sglang.srt.layers.moe.token_dispatcher.standard import (
            _map_topk_ids_to_local_experts,
        )
'''
assert candidate.count(unused)==1
candidate=candidate.replace(unused,'')
def owner(source):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xFp4Experts')
def members(source):
    return {n.name:n for n in owner(source).body if isinstance(n,ast.FunctionDef)}
b,c=members(baseline),members(candidate)
for name,node in b.items():
    if name not in ('__init__','_marlin_expert_outputs'):assert ast.dump(node)==ast.dump(c[name])
assert set(c)-set(b)=={'refresh_expert_lookup_','update_expert_map_'}
def assignment(node,name):
    return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
def cut(source):
    method=members(source)['_marlin_expert_outputs']
    start=next(i for i,n in enumerate(method.body) if assignment(n,'hidden_states'))
    end=next(i for i,n in enumerate(method.body) if assignment(n,'topk_weights'))
    return [n for n in method.body[start:end+1] if not isinstance(n,(ast.Import,ast.ImportFrom))]
bg=next(n for n in cut(baseline) if assignment(n,'fuse_mapping'))
cg=next(n for n in cut(candidate) if assignment(n,'fuse_mapping'))
assert ast.dump(bg)==ast.dump(cg)
# The entire method tail (real alignment/GEMMs/finalization) is unchanged.
def tail(source):
    body=members(source)['_marlin_expert_outputs'].body
    idx=next(i for i,n in enumerate(body) if assignment(n,'topk_weights'))
    return [ast.dump(n) for n in body[idx+1:]]
assert tail(baseline)==tail(candidate)
def snippet(source,node):
    lines=source.splitlines()
    start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
    return textwrap.dedent('\n'.join(lines[start-1:node.end_lineno]))
def adapter(source,name):
    ctor=members(source)['__init__'].body
    first=next(i for i,n in enumerate(ctor) if assignment(n,'expert_map'))
    last=next(i for i,n in enumerate(ctor) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='register_buffer' and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value==('_expert_lookup' if source==candidate else '_expert_map'))
    init='\n'.join(snippet(source,n) for n in ctor[first:last+1])
    prefix='\n'.join(snippet(source,n) for n in cut(source))
    result=f'class {name}(torch.nn.Module):\n    def __init__(self, num_experts=256, num_local_experts=32, local_expert_offset=64):\n        super().__init__()\n        self.intermediate_size_per_partition=32\n'+textwrap.indent(init,'        ')+'\n\n    def forward(self, hidden_states, topk_scores, topk_ids):\n'+textwrap.indent(prefix,'        ')+'\n        return fuse_mapping, local_topk_ids, topk_weights\n'
    if source==candidate:
        result+='\n'+textwrap.indent(snippet(source,members(source)['refresh_expert_lookup_']),'    ')+'\n\n'+textwrap.indent(snippet(source,members(source)['update_expert_map_']),'    ')+'\n'
    return result
helper=(ROOT/'profiling_kaggle/notebook04_support/mapping_helper.py').read_text(encoding='utf-8')
adapters='"""AST-selected init and pre-alignment statements; no real aligner/GEMM execution."""\nimport torch\nSMALL_NUMEL_LIMIT=64\n\n'+helper+'\n'+adapter(baseline,'BaselineSourceSlice')+'\n'+adapter(candidate,'CandidateSourceSlice')
ast.parse(candidate);ast.parse(adapters)
patch=''.join(difflib.unified_diff(baseline.splitlines(True),candidate.splitlines(True),fromfile='a/'+REL,tofile='b/'+REL))
(HERE/'cached_mapping_v4_draft.diff').write_text(patch,encoding='utf-8',newline='\n')
(HERE/'candidate_muse_spark_fp4_moe.py').write_text(candidate,encoding='utf-8',newline='\n')
(HERE/'source_slices.py').write_text(adapters,encoding='utf-8',newline='\n')
manifest={'baseline_patch_sha256':previous['patch_sha256'],'baseline_owner_sha256':previous['owner_source_sha256'],
          'candidate_source_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'draft_diff_sha256':hashlib.sha256(patch.encode()).hexdigest(),
          'source_slices_sha256':hashlib.sha256(adapters.encode()).hexdigest(),'mapping_helper_sha256':previous['mapping_helper_sha256'],
          'scope':'V4-relative draft NOT applied. Only selected constructor and pre-alignment statements execute in notebook. Full module and actual alignment/GEMMs untested.',
          'static_checks':{'other_methods_unchanged':True,'fused_mapping_guard_unchanged':True,'alignment_and_gemm_tail_unchanged':True}}
(HERE/'source_manifest_05.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Prepared unapplied draft and source-slice harness; preserved fused guard and full alignment/GEMM tail.')
