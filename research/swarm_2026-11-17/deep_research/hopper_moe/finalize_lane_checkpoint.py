from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
p=P/'write_deep_checkpoint.py';s=p.read_text(encoding='utf-8-sig').replace("'offline_small_k_repair']","'offline_small_k_repair','offline_w13_fused']");s=s.replace("'status':'active_research_checkpoint'","'status':'lane_checkpoint_complete_available_for_followup'")
s=s.replace('two SM90 jobs queued behind root hardware lease. Parent owns runtime evidence and integration.','two SM90 builds compile with zero spills and unchanged native MMA/launch/shared metadata. Parent owns runtime evidence and integration.')
s=s.replace("'next':'offline_w13_fused_sm90.py then Hopper numerical gates.'","'next':'Parent-owned full-chain numerical and performance evaluation; Hopper numerical qualification remains pending.'")
s=s.replace("'W13_FUSED_INDEPENDENT_REVIEW.json']","'W13_FUSED_INDEPENDENT_REVIEW.json','W13_FUSED_SM90_SASS_AUDIT.json']")
p.write_text(s,encoding='utf-8')
