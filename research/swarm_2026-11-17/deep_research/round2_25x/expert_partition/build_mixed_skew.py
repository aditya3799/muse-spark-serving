from pathlib import Path
import ast
p=Path(__file__).resolve().parent;s=(p/'probe_composed.py').read_text(encoding='utf-8-sig')
s=s.replace("pool=range(e) if pattern=='hot_rank' else range(e*8)","pool=range(e) if pattern=='hot_rank' else range(e*8)\n     if pattern=='mixed_skew' and token<m//2:pool=[(generation*8+j)%e for j in range(8)]")
s=s.replace("row=rng.sample(pool,8)", "row=rng.sample(pool,8)\n     assert len(set(row))==8")
s=s.replace("'COMPLETED_LOCAL_FOUR_WAY'", "'COMPLETED_LOCAL_FOUR_WAY_MIXED_SKEW'")
s=s.replace("'scope':'Local SM86 full MoE helper plus finalizer; synthetic E32 H4096, no serving/H200 claim'", "'scope':'Local SM86 mixed-skew four-way full MoE helper plus finalizer; half tokens route to eight hot local experts, half unique global top8, not representative serving data'")
(p/'probe_mixed_skew.py').write_text(s,encoding='utf-8');ast.parse(s);print('Prepared mixed-skew full four-way harness; no GPU started')
