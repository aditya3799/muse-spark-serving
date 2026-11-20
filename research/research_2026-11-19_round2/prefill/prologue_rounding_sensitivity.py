import json,math,struct
from pathlib import Path
h=Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/prefill')
j=json.loads((h/'prefill_prologue_diagnostic.json').read_text());x=j['diagnostics']['samples'][0]['input_head']
f=lambda a:struct.unpack('f',struct.pack('f',a))[0]
sq=[f(v*v) for v in x]
def tree(a):
 while len(a)>1:a=[f(a[i]+a[i+1]) for i in range(0,len(a),2)]
 return a[0]
def butterfly(a):
 while len(a)>1:
  n=len(a)//2;a=[f(a[i]+a[i+n]) for i in range(n)]
 return a[0]
sequential=0.
for v in sq:sequential=f(sequential+v)
result={'scope':'CPU FP32 reduction-order sensitivity of preserved exactBF16 input. Does NOT identify the actual GPU reduction tree or approximate rsqrt implementation.',
 'sum_double':sum(v*v for v in x),'sum_pair_adjacent':tree(sq),'sum_butterfly_halves':butterfly(sq),'sum_sequential':sequential,'examples':[]}
for name,total in [('double',sum(v*v for v in x)),('adjacent',tree(sq)),('halves',butterfly(sq)),('sequential',sequential)]:
 variance=f(f(total/64)+f(1e-5));out=f(x[4]*f(1/math.sqrt(variance)))
 result['examples'].append(dict(order=name,variance_plus_eps=variance,normalized_col4=out,bf16_boundary=(.58984375+.59375)/2,delta_from_boundary=out-(.58984375+.59375)/2))
(h/'prologue_rounding_sensitivity.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
