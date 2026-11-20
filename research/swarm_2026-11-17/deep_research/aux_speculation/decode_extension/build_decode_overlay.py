from pathlib import Path
import argparse,ast,difflib,hashlib,json
P=Path(__file__).resolve().parent;A=P.parent;ROOT=A.parents[2]
pa=argparse.ArgumentParser();pa.add_argument('--fused',action='store_true');args=pa.parse_args()
BASE=A/('fusion_store/integration_sources' if args.fused else 'candidate_sources');OUT=P/('fused_sources' if args.fused else 'direct_sources');records=[];delta=[]
def replace(s,old,new,n=1):
    assert s.count(old)==n,(old[:90],s.count(old))
    return s.replace(old,new)
def save(rel,base,s):
    if rel.endswith('.py'):ast.parse(s)
    dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(s)
    delta.extend(difflib.unified_diff((base or '').splitlines(True),s.splitlines(True),fromfile='a/'+rel if base else '/dev/null',tofile='b/'+rel))
    records.append({'path':rel,'base_sha256':hashlib.sha256(base.encode()).hexdigest() if base else None,'sha256':hashlib.sha256(s.encode()).hexdigest()})
for path in BASE.rglob('*'):
    if not path.is_file() or path.suffix not in ('.py','.cuh'):continue
    rel=path.relative_to(BASE).as_posix();base=path.read_text();s=base
    if rel.endswith('/models/muse_spark.py'):
        s=replace(s,'from sglang.srt.runtime_context import get_forward, get_memory, get_parallel','from sglang.srt.runtime_context import get_disagg, get_exec, get_forward, get_memory, get_parallel')
        anchor='    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:'
        methods='    def decode_aux_output_spec(self):\n        parallel = get_parallel()\n        if (not self.capture_aux_hidden_states or parallel.pp_size != 1\n            or parallel.attn_dp_enabled or parallel.attn_cp_size != 1\n            or get_exec().overlap.enable_two_batch_overlap\n            or get_disagg().enable_pdmux):\n            return None\n        return self.model.prefill_aux_output_spec()\n\n    def get_aux_hidden_states_width(self) -> int:\n        spec = self.decode_aux_output_spec()\n        return spec[0] * len(spec[1]) if spec is not None else 0\n\n'
        s=replace(s,anchor,methods+anchor)
    if rel.endswith('/speculative/dflash_worker_v2.py'):
        old='        runner = self.target_worker.model_runner.prefill_cuda_graph_runner\n        owner = getattr(runner, "prefill_aux_output_owner", None)\n'
        # Append chooses the exact storage owner; outer generation handles both.
        assert s.count(old)==2
        first='        owner = next(\n            (owner for owner in self._aux_output_owners()\n             if owner.owns(kwargs["target_hidden"])), None\n        )\n'
        s=s.replace(old,first,1)
        outer=old+'        generation = owner.generation if owner is not None else None\n'
        s=replace(s,outer,'        generations = [(owner, owner.generation) for owner in self._aux_output_owners()]\n')
        s=replace(s,'            if owner is not None:\n                owner.discard_ready(generation + 1)','            for owner, generation in generations:\n                owner.discard_ready(generation + 1)')
        anchor='    def _append_target_hidden_to_draft_kv_by_loc(self, **kwargs) -> None:\n'
        method='    def _aux_output_owners(self):\n        model_runner = self.target_worker.model_runner\n        owners = []\n        for runner_name, owner_name in (\n            ("prefill_cuda_graph_runner", "prefill_aux_output_owner"),\n            ("decode_cuda_graph_runner", "decode_aux_output_lease"),\n        ):\n            runner = getattr(model_runner, runner_name, None)\n            owner = getattr(runner, owner_name, None)\n            if owner is not None:\n                owners.append(owner)\n        return owners\n\n'
        s=replace(s,anchor,method+anchor)
    save(rel,base,s)
rel='python/sglang/srt/layers/decode_aux_output.py';save(rel,None,(P/'decode_aux_output.py').read_text())
rel='python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py';base=(ROOT/'sglang'/rel).read_text();s=base
s=replace(s,'from sglang.srt.layers.logits_processor import LogitsProcessorOutput','from sglang.srt.layers.logits_processor import LogitsProcessorOutput\nfrom sglang.srt.layers.decode_aux_output import DecodeAuxOutputLease')
s=replace(s,'        self.backend = resolve_decode_backend(self)','        self.backend = resolve_decode_backend(self)\n        self._init_decode_aux_output_lease()')
anchor='    def _next_token_logits_buffer_capacity_rows(self, max_num_tokens: int) -> int:\n'
methods='    decode_aux_output_lease = None\n\n    def _init_decode_aux_output_lease(self):\n        spec_fn = getattr(self.model_runner.model, "decode_aux_output_spec", None)\n        if (spec_fn is None or self.buffers.aux_hidden_states is None\n            or self.model_runner.is_draft_worker\n            or not self.model_runner.spec_algorithm.is_dflash()\n            or self.pp_size != 1 or self.enable_pdmux or self.enable_two_batch_overlap\n            or get_parallel().attn_dp_enabled or get_parallel().attn_cp_size != 1\n            or torch.device(self.device).type != "cuda"\n            or self.buffers.aux_hidden_states.device.type != "cuda"\n            or not self.capture_forward_mode.is_target_verify()):\n            return\n        spec = spec_fn()\n        if spec is not None:\n            self.decode_aux_output_lease = DecodeAuxOutputLease(\n                self.buffers.aux_hidden_states, spec[0], spec[1]\n            )\n\n    def _validate_decode_aux_binding(self):\n        lease = self.decode_aux_output_lease\n        if lease is not None:\n            lease.validate_binding(\n                self.buffers.aux_hidden_states,\n                self.model_runner.model.decode_aux_output_spec(),\n            )\n\n    def _decode_aux_capture_buffer(self, num_tokens):\n        if hasattr(self.model_runner.model, "decode_aux_output_spec"):\n            self._validate_decode_aux_binding()\n            lease = self.decode_aux_output_lease\n            return lease.view(num_tokens) if lease is not None else None\n        buffer = self.buffers.aux_hidden_states\n        return buffer[:num_tokens] if buffer is not None else None\n\n    def adopt_owned_cuda_graph_output(self, output, rows):\n        lease = self.decode_aux_output_lease\n        return lease.adopt(output, rows) if lease is not None else None\n\n'
s=replace(s,anchor,methods+anchor)
s=replace(s,'        aux_hidden_states_buffer = (\n            buffers.aux_hidden_states[:num_tokens]\n            if buffers.aux_hidden_states is not None\n            else None\n        )','        aux_hidden_states_buffer = self._decode_aux_capture_buffer(num_tokens)')
s=replace(s,'    def capture(self) -> None:\n        with self.model_context():\n            self._capture()\n','    def capture(self) -> None:\n        self._validate_decode_aux_binding()\n        lease = self.decode_aux_output_lease\n        generation = lease.begin_capture() if lease is not None else None\n        try:\n            with self.model_context():\n                self._capture()\n        finally:\n            if lease is not None:\n                lease.end_capture(generation, getattr(self, "stream", None))\n')
anchor='    def execute(\n        self,\n        forward_batch: ForwardBatch,\n        pp_proxy_tensors: Optional[PPProxyTensors] = None,\n    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:\n'
wrapper='    def execute(self, forward_batch, pp_proxy_tensors=None):\n        lease = self.decode_aux_output_lease\n        if lease is None:\n            return self._execute_with_aux_output(forward_batch, pp_proxy_tensors)\n        self._validate_decode_aux_binding()\n        if not forward_batch.forward_mode.is_target_verify() or forward_batch.batch_size <= 0:\n            raise RuntimeError("decode auxiliary lease requires nonempty target verification")\n        generation = lease.begin_produce()\n        try:\n            output = self._execute_with_aux_output(forward_batch, pp_proxy_tensors)\n            lease.end_produce(generation, output.hidden_states)\n            return output\n        except BaseException:\n            lease.cancel_produce(generation)\n            raise\n\n'+anchor.replace('def execute(', 'def _execute_with_aux_output(')
s=replace(s,anchor,wrapper);save(rel,base,s)
(P/('decode_fused_delta.diff' if args.fused else 'decode_direct_delta.diff')).write_text(''.join(delta))
(P/('fused_manifest.json' if args.fused else 'manifest.json')).write_text(json.dumps({'status':'CPU_PREPARED_UNVALIDATED','base':str(BASE),'files':records,'bytes':len(''.join(delta).encode())},indent=2)+'\n')
print('Prepared separate decode '+('fused' if args.fused else 'direct-copy')+' extension')
