"""Make a non-timed production integration and graph replay receipt."""
from pathlib import Path
here=Path(__file__).resolve().parent
s=(here/"paired_hold.py").read_text()
s=s.replace('    env=DenseEnvelope(m,args.seed)','    report.setdefault("plans",{})[str(m)]=split._choose_large_prefill_configs(layer,m*8,256)\n    save()\n    env=DenseEnvelope(m,args.seed)')
start=s.index('    for dist in ("normal","skewed"):\n',s.index('def case(m):'))
end=s.index('for m in args.rows:',start)
s=s[:start]+s[end:]
s=s.replace('report["complete"]=True;save()','report["correct"]=all(v.get("exact",False) for v in report["package_checks"]);report["complete"]=True;save();print(report["package_checks"],flush=True)')
(here/"paired_smoke.py").write_text(s)
launcher=(here/"launch_hold.py").read_text().replace('"paired_hold")','"paired_hold", "paired_smoke")')
(here/"launch_smoke.py").write_text(launcher)
