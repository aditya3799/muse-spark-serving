import json,random,struct,collections,math
from pathlib import Path
h=Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/prefill')
x=json.loads((h/'prefill_prologue_diagnostic.json').read_text())['diagnostics']['samples'][0]['input_head']
f=lambda a:struct.unpack('f',struct.pack('f',a))[0]
sq=[f(v*v) for v in x]
def tree(a):
 while len(a)>1:a=[f(a[i]+a[i+1]) for i in range(0,len(a),2)]
 return a[0]
rng=random.Random(261119);counts=collections.Counter()
for _ in range(4096):
 a=sq.copy();rng.shuffle(a);counts[tree(a)]+=1
out={'scope':'CPU permutation sensitivity only; permutations are not claims about native GPU layout. Helps distinguish possible rounding direction before IR audit.',
 'outcomes':[{'sum':total,'count':count,'variance_plus_eps':f(f(total/64)+f(1e-5)),
 'normalized_col4_using_correctly_rounded_sqrt':f(x[4]*f(1/math.sqrt(f(f(total/64)+f(1e-5)))))} for total,count in sorted(counts.items())]}
(h/'prologue_reduction_permutations.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
