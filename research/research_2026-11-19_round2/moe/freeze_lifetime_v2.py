import json,hashlib,datetime
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'finite_lifetime_v2'
review='''Independent review by decode_research: v2 closes the scan-before-certificate TOCTOU and stage-object ABA holes. Active enqueue depth rejects same-thread reentrant retirement. A canonical lease per resident lifetime prevents overlapping generation invalidation. In-place reload permanently disables that owner; a new prepared owner is required for a fresh certificate.
The guarantee remains conditional: every owner mutation, including .data and raw CUDA writes, must retire first; retirement must retire all referencing graphs and synchronize all model streams. No automatic detection of untracked writes is claimed. The host execution guard must surround graph replay enqueue, not merely the Python functions called during capture. Actual model-runner integration is absent, so this is not an enabled serving feature.
No further source blocker was identified for the disabled prototype. CPU21 negative cases pass. Scanner GPU99-case proof predates only the raw-device-context fix; packed-resident scanner numerical source is unchanged. No repeat GPU timing or kernel build is required by this guard-only revision.
'''
(out/'INDEPENDENT_REVIEW.txt').write_text(review)
files=[]
for p in sorted(out.iterdir()):
 if p.is_file() and p.name!='manifest.json':files.append(dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
manifest=dict(frozen_at_utc=datetime.datetime.utcnow().isoformat()+'Z',files=files,numerical_kernel_unchanged_from='../finite_frozen/manifest.json',enabled=False,scored_candidate=False,remaining='Actual model-owned graph/reload lifecycle integration and H200 wholekernel/full-serving validation')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest())
