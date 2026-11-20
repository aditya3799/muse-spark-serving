"""Unapplied single-owner variant; forward arithmetic remains identical to prior draft."""
import ast,difflib,hashlib,json,textwrap
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path(r'D:\OpenKernelChallenge\serving')
REL='python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'
baseline=(ROOT/'sglang'/REL).read_text(encoding='utf-8')
prior=(ROOT/'profiling_kaggle/notebook05_support/candidate_muse_spark_fp4_moe.py').read_text(encoding='utf-8')
old_manifest=json.loads((ROOT/'profiling_kaggle/notebook06_support/source_manifest_06.json').read_text())
assert hashlib.sha256(prior.encode()).hexdigest()==old_manifest['candidate_source_sha256']
assert hashlib.sha256((ROOT/'sglang'/REL).read_bytes()).hexdigest()==old_manifest['baseline_owner_sha256']
start=prior.index('    @torch.no_grad()\n    def refresh_expert_lookup_')
end=prior.index('    def _marlin_expert_outputs(',start)
replacement='''    @property
    def _expert_map(self) -> Optional[torch.Tensor]:
        """View of the single owning lookup; stays aliased after module moves."""
        if self._expert_lookup is None:
            return None
        return self._expert_lookup[1:]

    @torch.no_grad()
    def update_expert_map_(self, mapping: torch.Tensor) -> None:
        """One metadata copy; the mapping view and lookup share storage.

        Update before consumers on an ordered stream. Replacing/moving the
        owning buffer requires graph recapture. Map replacement is unsupported.
        """
        expert_map = self._expert_map
        if expert_map is None:
            raise ValueError("Mapping cannot be enabled without reconstructing the module")
        if (
            mapping.shape != expert_map.shape
            or mapping.dtype != torch.int32
            or mapping.device != expert_map.device
        ):
            raise ValueError("Expert map update must preserve shape, int32 dtype and device")
        expert_map.copy_(mapping)

'''
candidate=prior[:start]+replacement+prior[end:]
registration='        self.register_buffer("_expert_map", expert_map, persistent=False)\n'
assert candidate.count(registration)==1
candidate=candidate.replace(registration,'').replace('# Derived from the fixed expert map; not part of model checkpoints.','# One owning buffer; _expert_map is a view, not a second registration.')
def members(source):
    owner=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xFp4Experts')
    return {n.name:n for n in owner.body if isinstance(n,ast.FunctionDef)}
p,c=members(prior),members(candidate)
for name,node in p.items():
    if name not in ('__init__','update_expert_map_','refresh_expert_lookup_'):assert ast.dump(node)==ast.dump(c[name])
assert 'refresh_expert_lookup_' not in c and '_expert_map' in c
assert ast.dump(p['_marlin_expert_outputs'])==ast.dump(c['_marlin_expert_outputs'])
def assigned(node,name):return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
def snippet(node):
    start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
    return textwrap.dedent('\n'.join(candidate.splitlines()[start-1:node.end_lineno]))
ctor=c['__init__'].body
i=next(i for i,n in enumerate(ctor) if assigned(n,'expert_map'))
j=next(i for i,n in enumerate(ctor) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='register_buffer' and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value=='_expert_lookup')
init='\n'.join(snippet(n) for n in ctor[i:j+1])
body=c['_marlin_expert_outputs'].body
i=next(i for i,n in enumerate(body) if assigned(n,'hidden_states'))
j=next(i for i,n in enumerate(body) if assigned(n,'topk_weights'))
prefix='\n'.join(snippet(n) for n in body[i:j+1] if not isinstance(n,(ast.Import,ast.ImportFrom)))
shared='''from typing import Optional

class SharedSourceSlice(torch.nn.Module):
    def __init__(self,num_experts=256,num_local_experts=32,local_expert_offset=64):
        super().__init__()
        self.intermediate_size_per_partition=32
'''+textwrap.indent(init,'        ')+'\n\n'+textwrap.indent(snippet(c['_expert_map']),'    ')+'\n\n'+textwrap.indent(snippet(c['update_expert_map_']),'    ')+'''

    def forward(self,hidden_states,topk_scores,topk_ids):
'''+textwrap.indent(prefix,'        ')+'\n        return fuse_mapping,local_topk_ids,topk_weights\n'
previous_slices=(ROOT/'profiling_kaggle/notebook06_support/source_slices.py').read_text(encoding='utf-8')
slices=previous_slices+'\n'+shared
ast.parse(candidate);ast.parse(slices)
patch=''.join(difflib.unified_diff(baseline.splitlines(True),candidate.splitlines(True),fromfile='a/'+REL,tofile='b/'+REL))
for name,text in [('single_owner_muse_spark_fp4_moe.py',candidate),('single_owner_v4_draft.diff',patch),('source_slices.py',slices)]:
    (HERE/name).write_text(text,encoding='utf-8',newline='\n')
manifest={'baseline_patch_sha256':old_manifest['baseline_patch_sha256'],'baseline_owner_sha256':old_manifest['baseline_owner_sha256'],
    'two_buffer_source_sha256':old_manifest['candidate_source_sha256'],'single_owner_source_sha256':hashlib.sha256(candidate.encode()).hexdigest(),
    'single_owner_draft_sha256':hashlib.sha256(patch.encode()).hexdigest(),'source_slices_sha256':hashlib.sha256(slices.encode()).hexdigest(),
    'mapping_helper_sha256':old_manifest['mapping_helper_sha256'],
    'static_checks':{'forward_including_alignment_gemm_tail_identical_to_two_buffer_draft':True,'other_existing_methods_unchanged':True},
    'scope':'Unapplied second draft: map is a property view into lookup, rather than a separately registered buffer. Same mapping arithmetic as first draft. Source slices only; map view has storage offset 1 (four bytes), so full fused-kernel validation remains required.'}
(HERE/'source_manifest_07.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Prepared single-owner draft and three source-slice variants. Live V4 untouched.')
