from pathlib import Path
import subprocess,json,sys
here=Path(__file__).resolve().parent;root=here.parents[1]
tool=Path(sys.prefix)/'lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump'
rows=[]
for r in json.loads((here/'offline_cycles.json').read_text()):
 rec={'stage':r['stage']}
 for name,p in [('reference',root/r['source']/'kernel.sm90a.cubin'),('cycles',here/(r['stage']+'_cycles_sm90.cubin'))]:
  out=subprocess.check_output([str(tool),'--dump-resource-usage',str(p)],text=True);rec[name]=out
 rows.append(rec)
(here/'sm90_resources.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
