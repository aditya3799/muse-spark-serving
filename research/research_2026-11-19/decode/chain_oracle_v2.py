"""New four-block fixture revision with independently validated SM86 SWA origin.

The original frozen-receipt fixtures are never changed. This is not production
patch code: Hopper already consumes the local-window offset correctly.
"""
import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'research_2026-11-18_rl_prefill/decoder_chain'
sys.path.insert(0,str(ROOT/'research_2026-11-18_rl'))
from run_pair import telemetry

p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
p.add_argument('--cases',nargs='+',default=['1x74160','4x8192','16x4096'])
p.add_argument('--rounds',type=int,default=8);a=p.parse_args()
directory=HERE/f'chain_oracle_v2_{a.seed}';directory.mkdir(exist_ok=False)
source=(OLD/'bench.py').read_text()
def replace(old,new):
 global source
 assert source.count(old)==1,(old[:80],source.count(old))
 source=source.replace(old,new)
replace('def plan(st,window):','def plan(st,window,local_lengths=None):')
replace('cache_seqlens=st.l,qkv_dtype=op.b16,cu_seqlens_q=cuq,',
 'cache_seqlens=st.l if local_lengths is None else local_lengths,qkv_dtype=op.b16,cu_seqlens_q=cuq,')
replace('slide=plan(st,(left,0)) if cand else None', '''# New fixture: compensate for the SM86 consumer ignoring BlockMN n_offset.
            # The same graph-owned exact-origin table and lengths are used by all arms.
            origin=(st.l-n-left).clamp_min(0)
            local_lengths=st.l-origin
            columns=origin[:,None]+torch.arange(left+n,device=dev)[None,:]
            local_table=st.sw.gather(1,columns.long()).contiguous()
            slide=plan(st,(left,0),local_lengths) if cand else None''')
replace('cache_seqlens=st.l,\n                    cache_batch_idx=',
 'cache_seqlens=st.l if block.full else local_lengths,\n                    cache_batch_idx=')
replace('if block.full else st.sw,','if block.full else local_table,')
replace('candidate="packet+swa_schedule+grouped_at_supported_M128",',
 'candidate="packet+swa_schedule+grouped_at_supported_M128",fixture_revision="SM86 exact-origin SWA oracle v2; original native local-window consumer is defective",')
(directory/'executed.py').write_text(source)
(directory/'manifest.json').write_text(json.dumps(dict(
 old_source_sha256=hashlib.sha256((OLD/'bench.py').read_bytes()).hexdigest(),
 executed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
 oracle='attention_oracle_v2.json',
 scope='New local BF16 fixture baseline. Not original attention arithmetic, not a candidate serving patch, not H200 evidence.'),indent=2))
sys.path.insert(0,str(OLD))
sys.argv=[str(OLD/'bench.py'),'--seed',str(a.seed),'--cases',*a.cases,'--rounds',str(a.rounds),'--out',str(directory/'result.json')]
with (ROOT/'research_2026-11-18_rl/.gpu.lock').open('a+') as guard:
 fcntl.flock(guard,fcntl.LOCK_EX)
 with telemetry(directory):exec(compile(source,str(OLD/'bench.py'),'exec'),{'__name__':'__main__','__file__':str(OLD/'bench.py')})
