from pathlib import Path
import ast,json,hashlib
here=Path(__file__).resolve().parent
s=(here/'bench_hybrid_tiled.py').read_text(encoding='utf-8-sig')
needle="            samples={arm:[] for arm in ('baseline','candidate','control')}"
insert="""            # Predeclared control repair: six complete balanced cycles at this
            # shape before timed work; numerical kernels and dispatch unchanged.
            for warm in range(6):
                order=('baseline','candidate','control') if warm%2==0 else ('control','candidate','baseline')
                for arm in order:
                    if graph_mode:graphs[arm].replay()
                    else:run(arm)
            stream.synchronize()
"""
assert s.count(needle)==1;s=s.replace(needle,insert+needle)
s=s.replace("f'tiled_full_ffn_{args.seed}.json'","f'tiled_holdout_{args.seed}.json'")
s=s.replace("default=6)","default=8)")
ast.parse(s);p=here/'bench_hybrid_tiled_holdout.py';p.write_text(s,encoding='utf-8')
plan=dict(seed=930349,paired_rounds=8,warmup_complete_cycles=6,unchanged_numerical_source_sha256=hashlib.sha256((here/'wgs_bridge_tiled.py').read_bytes()).hexdigest(),holdout_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),predeclared=True,complete_ffn=True,duplicate_parent_required_per_pair_range=[0.95,1.05],conservative_speed_gate='min(median(baseline),median(control))/median(candidate)>1.10',stop='One holdout only. Failed control shapes stay unqualified; no repeat until a new justified change.',scope='Restoration component plus completeFFN, not serving speedup or original canonical parent win.')
(here/'tiled_holdout_plan.json').write_text(json.dumps(plan,indent=2))
p=here/'launch.py';s=p.read_text(encoding='utf-8-sig').replace("'bench_hybrid_tiled'","'bench_hybrid_tiled','bench_hybrid_tiled_holdout'");p.write_text(s,encoding='utf-8');print(json.dumps(plan,indent=2))
