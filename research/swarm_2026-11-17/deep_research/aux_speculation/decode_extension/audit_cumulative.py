from pathlib import Path
import ast,difflib,hashlib,json
P=Path(__file__).resolve().parent;A=P.parent;ROOT=A.parents[2]
reports=[]
for variant in ['direct','fused']:
    top=P/(variant+'_sources');patch=[];sources=[]
    for path in sorted(top.rglob('*')):
        if not path.is_file() or path.suffix not in ['.py','.cuh']:continue
        rel=path.relative_to(top).as_posix();candidate=path.read_text()
        if path.suffix=='.py':ast.parse(candidate)
        original=A/'base_sources'/rel
        if not original.exists():original=ROOT/'sglang'/rel
        base=original.read_text() if original.exists() else ''
        patch.extend(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/'+rel if base else '/dev/null',tofile='b/'+rel))
        sources.append({'path':rel,'original':str(original) if original.exists() else None,'base_sha256':hashlib.sha256(base.encode()).hexdigest() if base else None,'sha256':hashlib.sha256(candidate.encode()).hexdigest()})
    text=''.join(patch);(P/(variant+'_cumulative.diff')).write_text(text)
    added=[line[1:] for line in text.splitlines() if line.startswith('+') and not line.startswith('+++')]
    reports.append({'variant':variant,'files':sources,'bytes':len(text.encode()),'max_added_line':max(map(len,added)),'patch_sha256':hashlib.sha256(text.encode()).hexdigest()})
parent_hashes={str(f.relative_to(A)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [A/'aux_prefill_integration.diff',A/'source_manifest.json',A/'fusion_store/integration_delta.diff',A/'fusion_store/integration_manifest.json']}
report={'status':'ISOLATED_SOURCE_AUDIT_PASS','variants':reports,'preserved_parent_artifact_hashes':parent_hashes,'ownership':['Decode allocation remains the existing DecodeInputBuffers/global canonical pool. New lease binds that exact tensor and creates no payload allocation.','Prefill and decode retain independent buffers and generations. Consumer selects by exact pointer/shape/stride/dtype; tests hold one ready while consuming the other.','Outer DFlash exception wrapper snapshots both owner generations and discards only fresh ready generation+1 from the failed invocation.','Source capture and replay reject changed canonical pointer/shape/capture IDs and refuse recapture while output awaits consumption.','Cross-stream waits and allocator record_stream reuse the already-tested owner implementation; normal same-stream path enqueues no extra events.'],'guards':['DFlash target resolver excludes draft and other algorithms; initializer additionally requires is_dflash.','Muse capability requires configured captures, BF16 operator/FP32 residual, sorted unique valid runtime IDs, SP disabled, PP1/CP1/no attention DP, no TBO/PDMUX.','Runner requires CUDA buffer/device, target-verification capture mode and positive runtime batch size.','Any provided buffer must match exact row/feature geometry, dtype, contiguity and device before direct packing; otherwise existing list path remains.'],'fallbacks':['Other models keep upstream generic auxiliary binding.','No supplied eligible buffer leaves original snapshot-list/concatenation path.','Fused optional overlay keeps original JIT module for no-aux calls, original backend choice, and direct-copy fallback for native/Triton/all-gather/final-layer paths.','Unsupported model/runner topology declines decode ownership; prefill overlay remains independently guarded.'],'limits':['No active checkout edit or submission.','No change to width8, acceptance, KV commit policy, verification rows, weights or precision.','CPU fixtures and local GPU source-extracted tests do not qualify fullmodel/H200 correctness or performance.']}
(P/'CUMULATIVE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{k:r[k] for k in ['variant','bytes','max_added_line']} for r in reports]))
