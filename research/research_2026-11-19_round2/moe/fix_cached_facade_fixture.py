from pathlib import Path
p=Path(r'D:\OpenKernelChallenge\serving\research_2026-11-19_round2\moe')
for name in ['test_facade_memory.py','bench_hybrid_fallback.py']:
 f=p/name;s=f.read_text(encoding='utf-8-sig')
 s=s.replace('with cached_threshold_for_preparation(here):facade=workspace.prepare(owner)','facade=workspace.prepare(owner)  # Native default tables were already prepared above.')
 s=s.replace('with cached_threshold_for_preparation(here):workspace.prepare(owner)','workspace.prepare(owner)  # Reuses the already prepared native default tables.')
 f.write_text(s,encoding='utf-8')
