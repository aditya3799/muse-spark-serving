from pathlib import Path
import ast,json,hashlib
p=Path('/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/moe');root=p.parents[1]/'sglang'
rel=Path('python/sglang/srt/layers/muse_spark')
out=p/'serving_humming_v1';(out/rel).mkdir(parents=True,exist_ok=True)
source=(p/'submission_humming_fp32_v1.py').read_text(encoding='utf-8-sig')
funcs=source[source.index('def prepare_humming'):source.index('def prepare(weights):')]
funcs=funcs.replace('def prepare_humming(', 'def _prepare_humming_experts(').replace('def indexed(', 'def _humming_expert_outputs(')
q=(root/rel/'muse_spark_nvfp4_moe.py').read_text()
anchor='    def process_weights_after_loading(self, layer):\n'
method='''    def _process_weights_for_humming(self, layer):
        from types import SimpleNamespace

        raw = SimpleNamespace()
        for alias, native in self._MARLIN_PARAM_ALIASES:
            setattr(raw, alias, getattr(layer, native))
        # Match stock Marlin's gate/up global-scale collapse exactly.
        if raw.w13_weight_scale_2.dim() > 1:
            raw.w13_weight_scale_2 = raw.w13_weight_scale_2[:, 0].contiguous()
        layer._humming_layer = _prepare_humming_experts(raw, torch)
        # Only the canonical Humming packed tensors remain resident.
        for _, native in self._MARLIN_PARAM_ALIASES:
            delattr(layer, native)

'''
assert q.count(anchor)==1;q=q.replace(anchor,method+anchor)
anchor='        backend = getattr(self, "_moe_runner_backend", None)\n'
q=q.replace(anchor,anchor+'        if backend is not None and backend.is_humming():\n            return self._process_weights_for_humming(layer)\n')
q+='\n\n'+funcs
(out/rel/'muse_spark_nvfp4_moe.py').write_text(q)
f=(root/rel/'muse_spark_fp4_moe.py').read_text()
anchor='        self.use_marlin = self.moe_runner_backend.is_marlin()\n'
replacement='''        self.use_humming = self.moe_runner_backend.is_humming()
        # This flag also selects the BF16 activation/finalization path and
        # prevents early FP4 activation quantization in the parent model.
        self.use_marlin = self.moe_runner_backend.is_marlin() or self.use_humming
'''
assert f.count(anchor)==1;f=f.replace(anchor,replacement)
anchor='        """Project local routes; finalization skips undefined inactive slots."""\n'
branch='''        if self.use_humming:
            from sglang.srt.layers.muse_spark.muse_spark_nvfp4_moe import (
                _humming_expert_outputs,
            )

            local_ids = topk_ids.to(torch.int32)
            if self._expert_map is not None:
                local_ids = self._expert_lookup[local_ids.clamp_min(-1) + 1]
            return _humming_expert_outputs(
                self._humming_layer, hidden_states.contiguous(),
                local_ids.contiguous(), torch,
            )

'''
assert f.count(anchor)==1;f=f.replace(anchor,anchor+branch)
(out/rel/'muse_spark_fp4_moe.py').write_text(f)
manifest={'status':'EXPERIMENTAL_NOT_VALIDATED','flags':{'moe_runner_backend':'humming'},'files':{},'use_marlin_semantics':'True selects shared BF16 activation/norm path for Humming; preserves existing projection guards. Native stock Marlin remains unchanged when backend marlin selected.'}
for file in (out/rel).glob('*.py'):
 ast.parse(file.read_text());manifest['files'][str(file.relative_to(out))]=hashlib.sha256(file.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
