"""Derive canonical scoring input without modifying raw benchmark receipts."""
import json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
raw=HERE/'dflash_graph_913785/result.json'
doc=json.loads(raw.read_text());doc['agent']='decode_research'
canonical=HERE/'dflash_graph_913785/canonical.json';canonical.write_text(json.dumps(doc,indent=2))
subprocess.run([sys.executable,str(ROOT/'research_2026-11-18_rl/score_trials.py'),str(canonical),'--output',str(HERE/'dflash_graph_reward.json')],check=True)
