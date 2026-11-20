from pathlib import Path
import ast
p=Path(__file__).resolve().parent
s=(p/'probe_composed.py').read_text(encoding='utf-8-sig')
s=s.replace("for target in ('split','fused','composed'):", "for target in ('composed',):")
s=s.replace("seq=[('A',graphs['original']),('B',graphs[target])]", "seq=[('A',graphs['split']),('B',graphs[target])]")
s=s.replace('for rep in range(8):','for rep in range(16):')
s=s.replace("while time.monotonic()-warm<.25:", "while time.monotonic()-warm<.5:")
s=s.replace("'variant':target,'cache_sweep':cold", "'variant':target,'reference':'split','cache_sweep':cold")
s=s.replace("'scope':'Local SM86 full MoE helper plus finalizer; synthetic E32 H4096, no serving/H200 claim'", "'scope':'Direct split versus composed local SM86 full MoE helper plus finalizer; exactness also against scored original, synthetic E32 H4096, no serving/H200 claim'")
s=s.replace("'COMPLETED_LOCAL_FOUR_WAY'", "'COMPLETED_DIRECT_SPLIT_VS_COMPOSED'")
(p/'probe_direct_pair.py').write_text(s,encoding='utf-8');ast.parse(s)
print('Prepared direct split/composed paired harness; numerical gate remains four-way; no GPU execution')
