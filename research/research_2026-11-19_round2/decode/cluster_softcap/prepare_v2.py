import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent;out=here/'v2';out.mkdir(exist_ok=True)
old=(here/'gluon_prototype.py').read_text()
(here/'frontend_failure_v1.json').write_text(json.dumps(dict(complete=False,stage='Gluon argmax frontend',
 error='GluonSemantic.arange() missing required layout inside Triton _reduce_with_indices',
 native_compiled=True,gpu_execution=False,source_sha256=hashlib.sha256(old.encode()).hexdigest()),indent=2))
combine='''@g.jit
def native_argmax_combine(value1,index1,value2,index2):
    tie=value1==value2 and index1<index2
    gt=value1>value2 or tie
    return gl.where(gt,value1,value2),gl.where(gt,index1,index2)

'''
new=old.replace('@g.jit\ndef cluster_softcap',combine+'@g.jit\ndef cluster_softcap')
new=new.replace('values,indices=gl.max(x,axis=1,return_indices=True,return_indices_tie_break_left=True)',
 'values,indices=gl.reduce((x,gl.broadcast_to(local[None,:],(8,4096))),axis=1,combine_fn=native_argmax_combine)')
(out/'gluon_prototype.py').write_text(new)
compiler=(here/'compile_offline.py').read_text().replace('root=here.parents[2]','root=here.parents[3]').replace("here.parents[1]/'coordination", "here.parents[2]/'coordination")
(out/'compile_offline.py').write_text(compiler)
print('prepared explicit native tuple reducer in separatev2')
