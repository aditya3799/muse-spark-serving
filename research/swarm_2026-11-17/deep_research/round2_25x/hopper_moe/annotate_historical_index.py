from pathlib import Path
import json,hashlib
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round2_25x\hopper_moe');O=P.parents[1]/'hopper_moe';f=O/'DEEP_RESEARCH_CHECKPOINT.json';r=json.loads(f.read_text(encoding='utf-8-sig'))
if 'round2_writer_correction' not in r:
 backup=P/'HISTORICAL_CHECKPOINT_BEFORE_WRITER_ANNOTATION.json';backup.write_bytes(f.read_bytes());r['round2_writer_correction']={'status':'HOLD_WRITER_OWNERSHIP_BUG','historical_checkpoint_sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),'cause':'Canonical epilogue writer divided already-collapsed math threads by original K partitions again, suppressing required output writers. Old virtual builds/gate are not numerically usable.','new_artifacts':'round2_25x/hopper_moe/H200_CORRECTED_GATE_MANIFEST.json pins corrected cubins; native gate prepared but unexecuted.'}
 for item in r['strongest_followups']:
  if item['lead']=='Exact physical-K serialization into register banks':item['historical_next_before_correction']=item['next'];item['status']='HOLD_WRITER_OWNERSHIP_BUG';item['next']='Do not run historical virtual gate. Corrected round2 H200 gate is separately pinned and unexecuted.'
 f.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
print('Historical index annotated; raw build receipts/cubins unchanged')
