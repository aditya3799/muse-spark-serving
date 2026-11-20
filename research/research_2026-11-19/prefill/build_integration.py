"""Build an unapplied cumulative research patch over the frozen scored release."""
import argparse,ast,difflib,hashlib,json,subprocess,textwrap
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PIN='297e54fd105e974cdf401e5b4d20253a22c16f90'
BASE='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b'

def original(path):return subprocess.check_output(['git','-C',str(ROOT/'sglang'),'show',PIN+':'+path]).decode()
def once(source,before,after):
    assert source.count(before)==1,(before[:100],source.count(before))
    return source.replace(before,after)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-row',action='store_true');parser.add_argument('--enable-n',action='store_true');parser.add_argument('--name',default='integration_row');args=parser.parse_args()
    parent=(ROOT/'release_2026-11-18/candidate.diff').read_bytes();assert hashlib.sha256(parent).hexdigest()==BASE
    destination=HERE/args.name;destination.mkdir(exist_ok=True)
    worker_path='python/sglang/srt/speculative/dflash_worker_v2.py';model_path='python/sglang/srt/models/dflash.py'
    worker=original(worker_path);model=original(model_path)
    old='    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:'
    assert model.count(old)==2
    model=model.replace(old,'    def project_target_hidden(self, target_hidden: torch.Tensor, *, projection=None) -> torch.Tensor:',1)
    model=once(model,'        projected = self.fc(target_hidden)','        projected = (self.fc if projection is None else projection)(target_hidden)')
    worker=once(worker,'        extend_lens: Optional[torch.Tensor] = None,\n    ) -> None:\n        """Materialize target context features',
        '        extend_lens: Optional[torch.Tensor] = None,\n        prefill_projection: bool = False,\n    ) -> None:\n        """Materialize target context features')
    worker=once(worker,'            ctx_hidden = self.draft_model.project_target_hidden(target_hidden)',
        '            projection = getattr(self, "_prefill_row_projection" if prefill_projection else "_decode_n_projection", None)\n'
        '            if projection is None:\n'
        '                ctx_hidden = self.draft_model.project_target_hidden(target_hidden)\n'
        '            else:\n'
        '                ctx_hidden = self.draft_model.project_target_hidden(target_hidden, projection=projection)')
    worker=once(worker,'                positions=positions,\n                extend_lens=ctx_lens,',
        '                positions=positions,\n                extend_lens=ctx_lens,\n                prefill_projection=True,')
    worker=once(worker,'        borrow surfaces at startup rather than under traffic."""\n',
        '        borrow surfaces at startup rather than under traffic."""\n        self._initialize_projection_adapters()\n')
    helper='''
def _initialize_projection_adapters(self):
    # Scheduler invokes prewarm_sampling on the real forward stream, after
    # initial pool allocation/graph capture and before post-capture KV resize.
    if getattr(self, "_projection_adapters_initialized", False):
        return
    self._projection_adapters_initialized = True
    self._prefill_row_projection = None
    self._decode_n_projection = None
    ordinary_tp = not get_parallel().attn_dp_enabled and get_parallel().attn_cp_size == 1
    with draft_tp_context(self.draft_parallel_layout):
        from sglang.srt.speculative.dflash_projection_row import create_prefill_projection
        group = get_parallel().tp_group
        configuration = get_exec().graph.cuda_graph_config
        prefill_rows = list(configuration.prefill.bs or ())
        chunk = get_schedule().chunked_prefill_size
        if chunk is not None and chunk > 0:
            prefill_rows.append(int(chunk))
        self._prefill_row_projection = create_prefill_projection(
            self.draft_model, group, prefill_rows, ordinary_tp=ordinary_tp
        )
        # Keep the output-column diagnostic independently toggled. This
        # collective declaration prevents rank-local environment divergence.
        import inspect
        enabled = os.environ.get("SGLANG_EXPERIMENTAL_DFLASH_OUTPUT_N", "N_DEFAULT") == "1"
        supported_model = "projection" in inspect.signature(self.draft_model.project_target_hidden).parameters
        flags = [None] * group.world_size
        torch.distributed.all_gather_object(flags, (enabled, ordinary_tp, supported_model), group=group.cpu_group)
        if all(flag == (True, True, True) for flag in flags):
            from sglang.srt.speculative.dflash_projection_n import create_projection
            decode_batches = configuration.decode.bs
            if decode_batches is None:
                decode_batches = range(1, int(configuration.decode.max_bs or 1) + 1)
            decode_rows = sorted({int(bs) * int(self.block_size) for bs in decode_batches})
            self._decode_n_projection = create_projection(
                self.draft_model.fc, group, mode="qualified", rows=decode_rows, measure=True
            )
'''.replace('N_DEFAULT','1' if args.enable_n else '0')
    worker=once(worker,'    def prewarm_sampling(self) -> SamplingPrewarmResult:',textwrap.indent(textwrap.dedent(helper).strip()+'\n\n','    ')+'    def prewarm_sampling(self) -> SamplingPrewarmResult:')
    adapter=(HERE/'row_fc_adapter.py').read_text()
    adapter=once(adapter,"str(HERE/'row_plans.so')","str(_compiled_library())")
    runtime=(HERE/'row_fc_runtime_helpers.py').read_text()
    runtime=runtime.replace("os.environ.get('SGLANG_EXPERIMENTAL_DFLASH_ROW_FC','0')","os.environ.get('SGLANG_EXPERIMENTAL_DFLASH_ROW_FC','"+('1' if args.enable_row else '0')+"')")
    additions={'python/sglang/srt/speculative/dflash_projection_row.py':adapter+'\n'+runtime,
               'python/sglang/srt/speculative/dflash_row_fc.cc':(HERE/'row_plans.cc').read_text()}
    decode=HERE.parent/'decode/output_n_package'
    for path in decode.iterdir():
        if path.suffix in ('.py','.cc'):additions['python/sglang/srt/speculative/dflash_projection_n/'+path.name]=path.read_text()
    changed={worker_path:worker,model_path:model};delta=[];manifest=[]
    for path,source in {**changed,**additions}.items():
        if path.endswith('.py'):ast.parse(source)
        output=destination/'sources'/path;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(source,encoding='utf8')
        before=original(path) if path in changed else ''
        delta.extend(difflib.unified_diff(before.splitlines(keepends=True),source.splitlines(keepends=True),fromfile='a/'+path if before else '/dev/null',tofile='b/'+path))
        manifest.append(dict(path=path,sha256=hashlib.sha256(source.encode()).hexdigest()))
    diff=''.join(delta)
    (destination/'delta.diff').write_text(diff,encoding='utf8')
    cumulative=parent+b'\n'+diff.encode()
    (destination/'candidate.diff').write_bytes(cumulative)
    (destination/'manifest.json').write_text(json.dumps(dict(parent_sha256=BASE,pin=PIN,row_enabled=args.enable_row,n_enabled=args.enable_n,candidate_sha256=hashlib.sha256(cumulative).hexdigest(),files=manifest),indent=2)+'\n')
    print(destination,'files',len(manifest),'sha256',hashlib.sha256(cumulative).hexdigest())

if __name__=='__main__':main()
