from pathlib import Path
import ast,hashlib,json
p=Path(__file__).resolve().parent;root=p.parents[1]
rel=Path('python/sglang/srt/layers/muse_spark');out=p/'serving_humming_v1'
base=root/'swarm_2026-11-16/coordination/round2_projection_attention/candidate_sources'/rel/'muse_spark_fp4_moe.py'
assert hashlib.sha256(base.read_bytes()).hexdigest()=='f99e8cb70afdecace940c654d16a30668a62c9833568dc8d528de2b5716ad069'
old=(out/rel/base.name).read_text(encoding='utf-8-sig');s=base.read_text(encoding='utf-8-sig')
start=old.index('        if self.use_humming:',old.index('    def _marlin_expert_outputs('));end=old.index('        from sgl_kernel.scalar_type',start)
branch=old[start:end]
s=s.replace('        self.use_marlin = self.moe_runner_backend.is_marlin()\n','''        self.use_humming = self.moe_runner_backend.is_humming()
        # Shared BF16 activation/finalization semantics also disable early FP4
        # activation quantization and preserve the existing projection guards.
        self.use_marlin = self.moe_runner_backend.is_marlin() or self.use_humming
''')
anchor='        """Project local routes; finalization skips undefined inactive slots."""\n'
assert s.count(anchor)==1;s=s.replace(anchor,anchor+branch)
s=s.replace('device=self.gate_up_proj.device,','device=next(self.parameters()).device,')
ast.parse(s);(out/rel/base.name).write_text(s,encoding='utf-8')
file=out/rel/'muse_spark_nvfp4_moe.py';s=file.read_text(encoding='utf-8-sig');tree=ast.parse(s)
helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_prepare_humming_experts','_humming_expert_outputs')]
helpers[0].body.insert(0,ast.Expr(value=ast.Constant(value='Pack one canonical NVFP4 layout with FP8 block scales.\n\nThe caller validates aligned shapes, allowing repacking without redundant\nzero-padding copies. Global scales are applied after each BF16 GEMM output\nto preserve the stock Marlin rounding boundary. Tuning overrides are owned\nby this layer; Humming\'s cached defaults remain unchanged.')))
helpers[1].body.insert(0,ast.Expr(value=ast.Constant(value='Compute local routes with FP32 accumulation and no Stream-K reduction.\n\nFirst and final outputs alias one buffer as in the stock Marlin path.\nThe caller applies the original per-expert norm and weighted finalization.')))
new='\n\n\n'.join(ast.unparse(n) for n in helpers)+'\n'
s=s[:s.index('def _prepare_humming_experts(')]+new
assert max(map(len,s.splitlines()))<=400
file.write_text(s,encoding='utf-8')
manifest=json.loads((out/'manifest.json').read_text());manifest['marlin_fallback_base_sha256']=hashlib.sha256(base.read_bytes()).hexdigest();manifest['conversion']='Validated already_padded=True avoids redundant zero-pad clones; peak runtime validation pending'
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('compact32 fallback preserved; helpers formatted; maxline',max(map(len,s.splitlines())))
