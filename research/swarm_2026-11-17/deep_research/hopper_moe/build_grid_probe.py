"""Generate isolated grid probe from the previously validated fixture preparation."""
from pathlib import Path
import hashlib
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
source=ROOT/'swarm_2026-11-16/projection_communication/phase4_hour/profile_humming_stages.py'
s=source.read_text()
assert hashlib.sha256(source.read_bytes()).hexdigest()=='e9a81c84b87e6326896df4d63796e51b446dc70de115c516ed7f6764db960d60'
s=s.split('    lookup=raw._expert_lookup')[0]
s=s.replace("OWN=Path(__file__).resolve().parent;SWARM=OWN.parents[1];HERE=SWARM/'moe';ROOT=SWARM.parent", "OWN=Path(__file__).resolve().parent;ROOT=OWN.parents[2];SWARM=ROOT/'swarm_2026-11-16';HERE=SWARM/'moe'")
s=s.replace("if args.graph_body != 1:raise ValueError('Stage profiling requires graph body exactly 1')", "if args.graph_body not in (1,8):raise ValueError('Use body1 latency or body8 throughput')")
s += (P/'grid_probe_tail.py.source').read_text(encoding='utf-8-sig')
(P/'grid_probe.py').write_text(s)
print('grid_probe.py',hashlib.sha256(s.encode()).hexdigest())

