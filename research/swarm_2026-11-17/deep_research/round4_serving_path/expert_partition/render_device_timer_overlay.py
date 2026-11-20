import ast,difflib,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
H=Path(__file__).resolve().parent; ROOT=next(p for p in H.parents if (p/'sglang/python/sglang/srt').is_dir())
rel='python/sglang/srt/managers/scheduler_components/metrics_reporter.py'; src=ROOT/'sglang'/rel; original=src.read_text()
old='''            if dw is not None:
                if hasattr(dw, "draft_runner"):
'''
new='''            if dw is not None:
                # DFlash uses a plain TpModelWorker, not legacy draft runners.
                # Reuse the existing opt-in timer; do not add an event pair.
                if (
                    ENABLE_METRICS_DEVICE_TIMER
                    and type(self.scheduler.draft_worker).__name__ == "DFlashWorkerV2"
                ):
                    dw.model_runner.device_timer = timer
                if hasattr(dw, "draft_runner"):
'''
assert original.count(old)==1
changed=original.replace(old,new); assert changed.replace(new,old)==original; ast.parse(changed)
out=H/'device_timer_overlay'/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(changed)
(H/'device_timer_dflash_overlay.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel)))
method=next(n for n in ast.walk(ast.parse(changed)) if isinstance(n,ast.FunctionDef) and n.name=='_install_device_timer_on_runners')
checks=[]
for enabled in [False,True]:
 for dflash in [False,True]:
  for present in [False,True]:
   cls=type('DFlashWorkerV2' if dflash else 'OtherSpecWorker',(),{}); worker=cls(); draft=NS(device_timer=None); legacy=NS(device_timer=None); worker.draft_worker=NS(model_runner=draft,draft_runner=legacy,draft_runner_list=[])
   timer=object() if present else None;target=NS(device_timer=None);owner=NS(scheduler=NS(tp_worker=NS(model_runner=target),draft_worker=worker),forward_pass_device_timer=timer)
   ns={'ENABLE_METRICS_DEVICE_TIMER':enabled};exec(compile(ast.Module(body=[method],type_ignores=[]),'actual-timer-overlay','exec'),ns);ns['_install_device_timer_on_runners'](owner)
   assert target.device_timer is timer and legacy.device_timer is timer
   assert draft.device_timer is (timer if enabled and dflash and present else None)
   checks.append({'flag':enabled,'dflash':dflash,'timer_present':present})
report={'status':'INERT_CPU_VALIDATED','activation':'Only existing SGLANG_ENABLE_METRICS_DEVICE_TIMER flag. Default-off; no new configuration or active install.','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'overlay_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'checks':checks,'scope':'Attach the same existing timer to actual DFlash plain TpModelWorker.model_runner; preserve target and legacy wiring. Each runner keeps its original event scope.','limitations':['Source-extracted CPU wiring only, no runtime timing/overhead evidence','Timer category alone does not distinguish worker identity; a diagnostic observer should annotate execution scope','No new events are inserted into the code; enabling previously-unwired draft timer creates its existing event pair at draft execution','No overlapping nested wraps are supported; current worker execution is sequential','Default-flag-disabled FPM-only behavior remains unchanged']}
(H/'DEVICE_TIMER_DFLASH_OVERLAY_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(checks),report['overlay_sha256'])
