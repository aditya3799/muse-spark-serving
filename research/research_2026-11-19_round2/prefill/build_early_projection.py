"""Unapplied startup-order amendment, relative to frozen joint530e source."""
import ast, difflib, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
out=HERE/'early_projection_unapplied'
out.mkdir(exist_ok=True)
rel=Path('python/sglang/srt/managers/scheduler.py')
source=(ROOT/'sglang'/rel).read_text()
old='''        self.init_all_attention_backends()
        self.init_all_cuda_graphs()
'''
new='''        self.init_all_attention_backends()
        # Qualify optional draft projection before captured model arenas exist.
        # Reuse the real serving stream below; registration is never captured.
        model_runner = self.tp_worker.model_runner
        device_module = torch.get_device_module(model_runner.device)
        self.schedule_stream = None if use_mlx() else device_module.Stream(priority=0)
        forward_stream = (
            model_runner.forward_stream
            if self.enable_overlap or get_parallel().pp_size > 1 or use_mlx()
            else self.schedule_stream
        )
        initialize_projection = getattr(
            self.draft_worker, "_initialize_projection_adapters", None
        )
        if initialize_projection is not None:
            with device_module.stream(forward_stream):
                initialize_projection()
            device_module.synchronize()
        self.init_all_cuda_graphs()
'''
assert source.count(old)==1
changed=source.replace(old,new)
oldline='        self.schedule_stream = None if use_mlx() else device_module.Stream(priority=0)\n'
assert changed.count(oldline)==2
at=changed.index(oldline,changed.index('        self.init_all_cuda_graphs()'))
changed=changed[:at]+changed[at:].replace(oldline,'',1)
ast.parse(changed)
path=out/'sources'/rel
path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(changed)
diff=''.join(difflib.unified_diff(source.splitlines(True),changed.splitlines(True),fromfile='a/'+rel.as_posix(),tofile='b/'+rel.as_posix()))
(out/'incremental.diff').write_text(diff)
manifest=dict(scope='Unapplied source-order proposal relative to joint530e + pinned scheduler, NOT a submission and NOT combined memory proof',
    patch_sha256=hashlib.sha256(diff.encode()).hexdigest(),bytes=len(diff.encode()),
    scheduler_before_sha256=hashlib.sha256(source.encode()).hexdigest(),scheduler_after_sha256=hashlib.sha256(changed.encode()).hexdigest(),
    guards='Existing DFlash registration geometry/collective/precision gates unchanged. Startup overlap rejects speculative models upstream.',
    open_requirements=['Actual full server startup and TP8 validation','Shared memory ownership and runtime fallback budget','Native H200 activation/output gate','Post-capture configured bucket trimming cannot activate unqualified shapes'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
