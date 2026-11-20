from pathlib import Path
import ast
p=Path(__file__).resolve().parent
s=(p/'probe_composed.py').read_text(encoding='utf-8-sig')
s=s.replace("p.add_argument('--gpu',action='store_true');", "p.add_argument('--gpu',action='store_true');p.add_argument('--correctness-only',action='store_true');")
s=s.replace("ids.copy_(torch.tensor(routes,device='cuda',dtype=torch.int32))", """if pattern=='crossing':
     cutoff=composed.plan.get('cutoff',1)
     counts=((cutoff-1,cutoff+1,3),(cutoff+1,3,cutoff-1),(0,0,0))[generation%3]
     for token,row in enumerate(routes):
      row[:]=rng.sample(range(e,e*8),8)
      for expert,count in enumerate(counts):
       if token<count:row[expert]=expert
     assert all(len(set(row))==8 for row in routes)
     routes=[[v if 0<=v<e else -1 for v in row] for row in routes]
    ids.copy_(torch.tensor(routes,device='cuda',dtype=torch.int32))""")
s=s.replace("for cold in (False,True):\n    warm=", "for cold in (() if args.correctness_only else (False,True)):\n    warm=")
s=s.replace("set_inputs(generation)\n    for graph", "set_inputs(generation)\n    for value,final in outputs.values():value.fill_(37.)\n    for graph")
s=s.replace("report={'status':'RUNNING'", "report={'status':'RUNNING'")
(p/'probe_crossing.py').write_text(s,encoding='utf-8');ast.parse(s)
print('Prepared only: targeted cutoff crossing with poisoned outputs; no GPU started')
