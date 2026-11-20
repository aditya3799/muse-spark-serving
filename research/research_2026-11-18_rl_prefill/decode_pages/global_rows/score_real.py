"""Score the two independent mechanisms on the matched real-layout receipts."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[2]
p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,default=H/'real_pilot.json');p.add_argument('--holdout',type=Path,default=H/'real_holdout.json');a=p.parse_args()
for arm,which,candidate,label in [('window','suffix',H.parent/'candidate.diff','Window-aware FA3 page translation with materialized full table'),('candidate','direct',H/'candidate.diff','Native FA3 row map with suffix-only SWA table')]:
    paths=[]
    for fname in (a.pilot,a.holdout):
        if not fname.exists():continue
        raw=json.loads(fname.read_text())
        if not raw['complete']:continue
        states=raw['stream_ownership'];assert states==dict(pre_event_preserved=False,post_event_preserved=True)
        record={k:v for k,v in raw.items() if k not in ('bench',)}
        record.update(candidate=label,candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),raw_trial=fname.name,raw_sha256=hashlib.sha256(fname.read_bytes()).hexdigest(),scope='Once-per-forward target FA3 metadata on scored dtype/capacity/disjoint layouts; native forced kernel RTX replay. Core and scheduler overlap excluded.')
        shapes={}
        for name,r in raw['bench']['shapes'].items():
            times=r['times'];shapes[name]=dict(weight=1,times={k:times[arm if k=='candidate' else k] for k in ('baseline','candidate','control')})
        record['bench']=dict(instances=1,shapes=shapes)
        dest=H/(fname.stem+'_'+which+'.json');dest.write_text(json.dumps(record,indent=2));paths.append(dest)
    if not paths:continue
    cmd=[sys.executable,str(ROOT/'research_2026-11-18_rl/score_trials.py'),str(paths[0]),'--output',str(H/(which+'_reward.json'))]
    if len(paths)==2:cmd+=['--holdout',str(paths[1])]
    subprocess.run(cmd,check=True)

