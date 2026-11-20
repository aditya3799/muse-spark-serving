"""Build an isolated source overlay; never modify the active checkout."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
SGL = ROOT / 'sglang'
SCORED = ROOT / 'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'
OUT = P / 'candidate_sources'
BASE = P / 'base_sources'
records = []
patch = []


def replace(s, before, after):
    assert s.count(before) == 1, (before[:100], s.count(before))
    return s.replace(before, after)


def save(rel, base, out, origin):
    ast.parse(out)
    dest = OUT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(out, encoding='utf-8')
    if base is not None:
        dest = BASE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(base, encoding='utf-8')
    patch.extend(difflib.unified_diff(
        (base or '').splitlines(True), out.splitlines(True),
        fromfile='a/' + rel if base is not None else '/dev/null', tofile='b/' + rel))
    records.append({'file': rel, 'origin': str(origin),
        'base_sha256': hashlib.sha256(base.encode()).hexdigest() if base else None,
        'candidate_sha256': hashlib.sha256(out.encode()).hexdigest()})

rel = 'python/sglang/srt/layers/prefill_aux_output.py'
save(rel, None, (P / 'prefill_aux_output.py').read_text(encoding='utf-8-sig'), 'new')

rel = 'python/sglang/srt/models/muse_spark.py'
base = (SCORED / rel).read_text(encoding='utf-8')
s = replace(base, 'from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states\n',
    'from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states\n'
    'from sglang.srt.layers.prefill_aux_output import direct_aux_capture\n')
spec = '''    def prefill_aux_output_spec(self):
        ids = tuple(self.layers_to_capture)
        if (
            self.operator_dtype != torch.bfloat16
            or self.residual_dtype != torch.float32
            or layernorm_sp.layernorm_sp_enabled()
            or not ids
            or any(type(i) is not int or not 0 <= i < len(self.layers) for i in ids)
            or ids != tuple(sorted(set(ids)))
        ):
            return None
        return self.config.hidden_size, ids

'''
anchor = '        # Layer indices whose output residual stream feeds the DFLASH draft context.\n        self.layers_to_capture: List[int] = []\n\n'
s = replace(s, anchor, anchor + spec)
s = replace(s, '        aux_hidden_states = []\n        for layer_idx, layer in enumerate(self.layers):',
'''        packed_aux = direct_aux_capture(
            (forward_batch.aux_hidden_states_buffer
             if forward_batch.forward_mode.is_extend() else None),
            stream,
            self.layers_to_capture,
            self.operator_dtype,
            sp_active,
        )
        aux_hidden_states = []
        for layer_idx, layer in enumerate(self.layers):''')
s = replace(s, '                aux_hidden_states.append(stream.to(self.operator_dtype))',
'''                if packed_aux is not None:
                    packed_aux.capture(layer_idx, stream)
                else:
                    aux_hidden_states.append(stream.to(self.operator_dtype))''')
s = replace(s, '        if aux_hidden_states:\n            return hidden_states, aux_hidden_states',
'''        if packed_aux is not None:
            return hidden_states, packed_aux.finalize()
        if aux_hidden_states:
            return hidden_states, aux_hidden_states''')
s = replace(s, '    @property\n    def layers(self):\n        return self.language_model.layers\n',
'''    def prefill_aux_output_spec(self):
        return self.language_model.prefill_aux_output_spec()

    @property
    def layers(self):
        return self.language_model.layers
''')
save(rel, base, s, SCORED / rel)

rel = 'python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py'
base = (SGL / rel).read_text(encoding='utf-8')
s = replace(base, 'from sglang.srt.layers.pooler import EmbeddingPoolerOutput\n',
    'from sglang.srt.layers.pooler import EmbeddingPoolerOutput\n'
    'from sglang.srt.layers.prefill_aux_output import PrefillAuxOutputOwner\n')
s = replace(s, '        self.layer_model = None\n',
    '        self.layer_model = None\n        self.prefill_aux_output_owner = None\n')
s = replace(s, '        # --- aiter chip info pre-warming (AMD) -------------------------\n',
'''        # Restrict shared output to a known synchronous DFlash consumer. Other
        # speculative algorithms and sharded/TBO paths retain existing outputs.
        self._init_prefill_aux_output_owner()

        # --- aiter chip info pre-warming (AMD) -------------------------
''')
methods = '''    def _init_prefill_aux_output_owner(self):
        spec_fn = getattr(self.layer_model, "prefill_aux_output_spec", None)
        if (
            not isinstance(self.backend, BreakableCudaGraphBackend)
            or self.model_runner.is_draft_worker
            or not self.model_runner.spec_algorithm.is_dflash()
            or self.pp_size != 1
            or get_parallel().attn_dp_enabled
            or self.enable_cp_bcg_capture
            or get_exec().overlap.enable_two_batch_overlap
            or torch.device(self.device).type != "cuda"
            or spec_fn is None
        ):
            return
        spec = spec_fn()
        if spec is not None:
            hidden_size, ids = spec
            self.prefill_aux_output_owner = PrefillAuxOutputOwner(
                self.max_num_tokens, hidden_size, ids, self.device
            )

    def _prefill_aux_output_view(self, num_tokens):
        owner = self.prefill_aux_output_owner
        return owner.view(num_tokens) if owner is not None else None

    def adopt_owned_cuda_graph_output(self, output, rows):
        owner = self.prefill_aux_output_owner
        return owner.adopt(output, rows) if owner is not None else None

'''
s = replace(s, '    def _is_mamba_track_enabled(self) -> bool:\n', methods + '    def _is_mamba_track_enabled(self) -> bool:\n')
s = replace(s, '                capture_hidden_mode=self.capture_hidden_mode,\n',
'''                capture_hidden_mode=self.capture_hidden_mode,
                aux_hidden_states_buffer=self._prefill_aux_output_view(num_tokens),
''')
s = replace(s, '            spec_info=padded_spec_info,\n            capture_hidden_mode=forward_batch.capture_hidden_mode,\n',
'''            spec_info=padded_spec_info,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            aux_hidden_states_buffer=self._prefill_aux_output_view(static_num_tokens),
''')
anchor = '''    def execute(
        self, forward_batch: ForwardBatch, **kwargs
    ) -> Optional[Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]]:
'''
wrapper = '''    def execute(
        self, forward_batch: ForwardBatch, **kwargs
    ) -> Optional[Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]]:
        owner = self.prefill_aux_output_owner
        if owner is None:
            return self._execute_with_aux_output(forward_batch, **kwargs)
        generation = owner.begin_produce()
        try:
            output = self._execute_with_aux_output(forward_batch, **kwargs)
            owner.end_produce(generation, output.hidden_states)
            return output
        except BaseException:
            owner.cancel_produce(generation)
            raise

    def _execute_with_aux_output(
        self, forward_batch: ForwardBatch, **kwargs
    ) -> Optional[Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]]:
'''
s = replace(s, anchor, wrapper)
s = replace(s, "    def capture(self) -> None:\n        with self.model_context():\n            self._capture()\n",
'''    def capture(self) -> None:
        owner = self.prefill_aux_output_owner
        generation = owner.begin_capture() if owner is not None else None
        try:
            with self.model_context():
                self._capture()
        finally:
            if owner is not None:
                owner.end_capture(generation, getattr(self, "stream", None))
''')
save(rel, base, s, SGL / rel)

rel = 'python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py'
base = (SGL / rel).read_text(encoding='utf-8')
s = replace(base, '        if torch.is_tensor(output):\n            return output.new_empty((size, *output.shape[1:]))',
'''        if torch.is_tensor(output):
            adopt = getattr(self._cuda_graph_runner, "adopt_owned_cuda_graph_output", None)
            owned = adopt(output, size) if adopt is not None else None
            if owned is not None:
                return owned
            return output.new_empty((size, *output.shape[1:]))''')
s = replace(s, '        if torch.is_tensor(output) and torch.is_tensor(output_buffer):\n            output_buffer[:num_tokens].copy_(output[:num_tokens])',
'''        if torch.is_tensor(output) and torch.is_tensor(output_buffer):
            adopt = getattr(self._cuda_graph_runner, "adopt_owned_cuda_graph_output", None)
            owned = adopt(output, num_tokens) if adopt is not None else None
            if (
                owned is not None
                and output_buffer.data_ptr() == owned.data_ptr()
                and output_buffer.stride() == owned.stride()
                and output_buffer.dtype == owned.dtype
                and output_buffer.device == owned.device
                and output_buffer.shape[1:] == owned.shape[1:]
                and output_buffer.shape[0] >= num_tokens
            ):
                return
            output_buffer[:num_tokens].copy_(output[:num_tokens])''')
save(rel, base, s, SGL / rel)

rel = 'python/sglang/srt/speculative/dflash_worker_v2.py'
base = (SGL / rel).read_text(encoding='utf-8')
s = replace(base, 'import os\n', 'import os\nfrom contextlib import nullcontext\n')
anchor = '    def _append_target_hidden_to_draft_kv_by_loc(\n'
wrapper = '''    def _append_target_hidden_to_draft_kv_by_loc(self, **kwargs) -> None:
        runner = self.target_worker.model_runner.prefill_cuda_graph_runner
        owner = getattr(runner, "prefill_aux_output_owner", None)
        lease = (
            owner.consume(kwargs["target_hidden"])
            if owner is not None
            else nullcontext()
        )
        # Release follows enqueue of the context projection and every KV write.
        # A later producer stream waits for these reads before overwriting.
        with lease:
            self._append_target_hidden_to_draft_kv_by_loc_impl(**kwargs)

    def _append_target_hidden_to_draft_kv_by_loc_impl(
'''
s = replace(s, anchor, wrapper)
anchor = '    def forward_batch_generation(\n'
wrapper = '    def forward_batch_generation(self, *args, **kwargs):\n        runner = self.target_worker.model_runner.prefill_cuda_graph_runner\n        owner = getattr(runner, "prefill_aux_output_owner", None)\n        generation = owner.generation if owner is not None else None\n        try:\n            return self._forward_batch_generation_with_aux_output(*args, **kwargs)\n        except Exception:\n            if owner is not None:\n                owner.discard_ready(generation + 1)\n            raise\n\n    def _forward_batch_generation_with_aux_output(\n'
s = replace(s, anchor, wrapper)
save(rel, base, s, SGL / rel)

(P / 'aux_prefill_integration.diff').write_text(''.join(patch), encoding='utf-8')
(P / 'source_manifest.json').write_text(json.dumps({
    'status':'ISOLATED_UNQUALIFIED_OVERLAY', 'no_submission':True,
    'base_model':'scored round5 candidate', 'other_base':'pinned active checkout, unchanged files',
    'files':records, 'patch_bytes':len(''.join(patch).encode()),
    'max_added_line':max(len(l) - 1 for l in patch if l.startswith('+')),
}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'files':len(records),'patch_bytes':len(''.join(patch).encode())}))
