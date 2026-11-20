"""Reconstruct original SM86 PTX reduction tree for the preserved BF16 row."""
import hashlib,json,struct
from pathlib import Path
H=Path(__file__).resolve().parent;O=H/'prologue_rounding_debug'
f=lambda a:struct.unpack('f',struct.pack('f',a))[0]
x=json.loads((H/'prefill_prologue_diagnostic.json').read_text())['diagnostics']['samples'][0]['input_head']
def local(a):
 # BF16 squares fit FP32 exactly here; PTX pair fma(x0,x0,x1*x1)
 # therefore equals one rounded addition of exact products.
 a=[f(a[i]*a[i]+f(a[i+1]*a[i+1])) for i in range(0,len(a),2)]
 while len(a)>1:a=[f(a[i]+a[i+1]) for i in range(0,len(a),2)]
 return a[0]
def reduce(per_thread):
 lanes=[local(x[i:i+per_thread]) for i in range(0,64,per_thread)]
 offset=len(lanes)//2
 while offset:
  lanes=[f(v+lanes[i^offset]) for i,v in enumerate(lanes)];offset//=2
 return lanes[0]
receipt=json.loads((O/'reproducer.json').read_text())['cases'][1]
rows=[]
for label,per_thread in [('native',8),('candidate',16)]:
 total=reduce(per_thread);variance=f(total/64)
 observed=receipt['debug'][label]['variance']
 assert variance==observed,(label,variance,observed)
 source=O/(label+'_original_m4095.ttgir');ptx=O/(label+'_original_m4095.ptx')
 rows.append(dict(label=label,per_thread_columns=per_thread,lanes_per_head=64//per_thread,
  shuffle_offsets=[4,2,1] if per_thread==8 else [2,1],cpu_sum=total,cpu_variance=variance,gpu_observed_variance=observed,
  exactly_reconstructed=True,ttgir_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),ptx_sha256=hashlib.sha256(ptx.read_bytes()).hexdigest()))
out=dict(scope=__doc__,passed=True,rows=rows,
 conclusion='The exact changed per-thread grouping and butterfly sequence reconstruct both GPU variances. BF16/FP8 boundary preservation does not preserve input reduction bits when compiler layout changes.')
(O/'reduction_tree_proof.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
