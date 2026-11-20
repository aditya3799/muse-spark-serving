import ast,difflib,hashlib,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent;V1=H/'target_argmax_overlay/python';V2=H/'target_argmax_overlay_v2/python'
core='sglang/srt/speculative/dflash_target_argmax_capture.py';adapter='sglang/srt/speculative/dflash_target_argmax_adapter.py'
changes={core:[('or raw_rows != captured_rows or width != self.width', 'or type(raw_rows) is not int or type(captured_rows) is not int\n                or not (0 < raw_rows <= captured_rows)\n                or raw_rows % self.width or width != self.width'),('# Exact-bucket admission avoids silently consuming padded/stale rows.','# Plain fixed-width replay preserves the leading raw token rows.\n        # Padded rows are computed, but only the live prefix is consumed.')],adapter:[('or runner.raw_num_token != raw_rows','or runner.raw_num_token != raw_rows\n                or type(runner.raw_bs) is not int or runner.raw_bs <= 0\n                or runner.raw_bs > runner.bs\n                or runner.raw_bs * runner.captured_req_width != raw_rows')]}
files=[];diff=[]
for src in V1.rglob('*.py'):
 rel=src.relative_to(V1).as_posix();old=src.read_text();new=old
 for before,after in changes.get(rel,[]):
  assert new.count(before)==1,(rel,before);new=new.replace(before,after)
 ast.parse(new);dest=V2/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(new)
 files.append({'path':rel,'v1_file_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'v2_file_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
 diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='v1/'+rel,tofile='v2/'+rel))
(H/'target_argmax_padded_v2_incremental.diff').write_text(''.join(diff))
(H/'TARGET_ARGMAX_PADDED_V2_MANIFEST.json').write_text(json.dumps({'status':'SEPARATE_DEFAULT_OFF_PADDED_ADMISSION_CPU_ONLY','files':files,'preserved':'v1 source/overlay/results untouched','scope':'Same noDP/CP/PP/LoRA/ragged/variant/stream/LP guards, plain fixed width8, actual runner raw_bs*W==raw_num_token==raw_rows and0<raw_rows<=captured_rows; consume leading output[:raw_rows].','held':['native padded poison test','real TP8 NCCL and model integration','graph pool borrowing and memory-saver','serving performance']},indent=2)+'\n')
# Keep all prior lifecycle negatives, adding actual raw_bs provenance and padded positives.
s=(H/'test_target_argmax_overlay.py').read_text();s=s.replace("O=H/'target_argmax_overlay/python/sglang/srt'", "O=H/'target_argmax_overlay_v2/python/sglang/srt'")
s=s.replace('runner.bs=2; runner.raw_num_token=16', 'runner.bs=2; runner.raw_bs=2; runner.raw_num_token=16')
s=s.replace("old=r.backend._graphs[Key()]", '''r.raw_bs=1; r.raw_num_token=8
ok('padded live prefix accepted',consume(a,raw_rows=8).numel()==8)
for raw in [0,1,7,9,17,24]: ok('padded invalid row boundary '+str(raw),consume(a,raw_rows=raw) is None)
r.raw_bs=2; ok('raw_bs disagreement rejects prefix',consume(a,raw_rows=8) is None)
r.raw_bs=0; ok('empty raw batch rejects',consume(a,raw_rows=8) is None)
r.raw_bs=3; r.raw_num_token=24; ok('raw batch exceeds bucket rejects',consume(a,raw_rows=24) is None)
r.raw_bs=2; r.raw_num_token=16
old=r.backend._graphs[Key()]''')
s=s.replace("'TARGET_ARGMAX_OVERLAY_CPU_RESULT.json'", "'TARGET_ARGMAX_PADDED_V2_CPU_RESULT.json'")
(H/'test_target_argmax_padded_v2.py').write_text(s)
print('Prepared separate V2',len(files))
