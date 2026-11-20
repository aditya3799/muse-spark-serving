from pathlib import Path
import hashlib,json,shutil
root=Path('/mnt/d/OpenKernelChallenge/serving')
swarm=root/'swarm_2026-11-16'
out=swarm/'coordination/round1_row'
dest=out/'candidate_sources'
base=root/'optimization_marlin_tiles_2026-11-15/candidate_compact32_01/candidate_sources'
projection=swarm/'projection_communication/candidate_sources'
model='python/sglang/srt/models/muse_spark.py'
assert (base/model).read_bytes()==(root/'sglang'/model).read_bytes(), 'baseline model conflict'
records={}
for tree,label in ((base,'scored_compact32'),(projection,'row_projection')):
    for p in sorted(tree.rglob('*.py')):
        rel=p.relative_to(tree)
        if '__pycache__' in rel.parts:continue
        d=dest/rel;d.parent.mkdir(parents=True,exist_ok=True)
        d.write_bytes(p.read_bytes())
        records[rel.as_posix()]={'source':label,'sha256':hashlib.sha256(d.read_bytes()).hexdigest()}
(out/'integration_manifest.json').write_text(json.dumps({'parents':['468701af35b84e07','projection_communication row candidate'],'files':records,'status':'experimental_integrated_not_submitted','no_active_checkout_changes':True},indent=2)+'\n')
print(json.dumps({'out':str(out),'files':len(records),'status':'assembled'}))
