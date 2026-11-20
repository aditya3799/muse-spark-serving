import pathlib
p=pathlib.Path('serving/research_2026-11-19_round2/moe/pr50800_offline/compile.py');s=p.read_text();old="assert free['MemAvailable']>=int(2.3*1024**3),('host memory guard',free['MemAvailable'])"
new="""reads=[];stable=0;deadline=time.monotonic()+60
while time.monotonic()<deadline:
 free={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
 reads.append(dict(monotonic=time.monotonic(),available=free['MemAvailable']))
 stable=stable+1 if free['MemAvailable']>=int(2.3*1024**3) else 0
 if stable>=3:break
 time.sleep(1)
(here/'stability_preflight.json').write_text(json.dumps(dict(reads=reads,stable=stable,required_available=int(2.3*1024**3),bounded_seconds=60),indent=2))
assert stable>=3,('host memory guard',reads[-1])"""
assert old in s;s=s.replace(old,new);p.write_text(s)
