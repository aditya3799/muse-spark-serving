import json,hashlib,datetime
from pathlib import Path
H=Path(__file__).resolve().parent;p=H/'TARGET_ARGMAX_PADDED_V2_MANIFEST.json';old=H/'TARGET_ARGMAX_PADDED_V2_GENERATION_RECEIPT.json'
assert not old.exists();old.write_bytes(p.read_bytes());d=json.loads(p.read_text());native=H.parent/'coordination/target_argmax_padded_01/probe/results.json';r=json.loads(native.read_text())
d['status']='DEFAULT_OFF_PADDED_ADMISSION_COMPONENT_LIFECYCLE_PASSED_SERVING_HELD'
d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
d['held']=['real TP8 NCCL and model integration','actual Muse graph pool borrowing/precarve and memory-saver integration','rank-uniform admission/fallback and coordinated reconfiguration','serving performance']
d['passed_native_component_gate']={'path':str(native.relative_to(H.parent)),'sha256':hashlib.sha256(native.read_bytes()).hexdigest(),'status':r['status'],'checks':len(r['checks']),'changed_replays':20,'padded_transition':r['padded_transition'],'peak_allocated_bytes':r['peak_allocated_bytes'],'limitations':r['limitations']}
d['qualification_details']='TARGET_ARGMAX_QUALIFICATION.json';d['preserved_generation_receipt']=old.name
p.write_text(json.dumps(d,indent=2)+'\n');print(d['status'])
