"""Offline-only SM90 prototype; no runtime wrapper or production activation."""
import triton.experimental.gluon as g
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.extra import libdevice
from triton.experimental.gluon.language.nvidia.hopper import cluster

@g.jit
def cluster_softcap(LOGITS, OUT, STRIDE, WIDTH, OFFSET, CAP):
    # Eight original4096-column tiles, one per CTA; each CTA retains8warps.
    layout:gl.constexpr=gl.BlockedLayout([1,1],[1,32],[1,8],[1,0],cga_layout=[[1,0],[2,0],[4,0]])
    tiles=gl.arange(0,8,layout=gl.SliceLayout(1,layout))
    local=gl.arange(0,4096,layout=gl.SliceLayout(0,layout))
    cols=tiles[:,None]*4096+local[None,:]
    row=gl.program_id(0).to(gl.int64)
    valid=cols<WIDTH
    x=gl.load(LOGITS+row*STRIDE+cols,valid,other=0.0)
    x=x/CAP
    x=libdevice.tanh(x)
    x=x*CAP
    x=gl.where(valid,x,-float('inf'))
    values,indices=gl.max(x,axis=1,return_indices=True,return_indices_tie_break_left=True)
    indices=indices+tiles*4096
    # Distribute the8 tile scalars within each CTA; compiler implements DSM
    # conversion and its synchronization, rather than hand-written PTX fences.
    replicated:gl.constexpr=gl.BlockedLayout([1],[32],[8],[0],cga_layout=[[0],[0],[0]])
    cluster.barrier()
    values=gl.convert_layout(values,replicated)
    indices=gl.convert_layout(indices,replicated)
    ranks=gl.arange(0,8,layout=replicated)
    best=-float('inf');best_index=0
    for tile in gl.static_range(8):
        # Integer extraction preserves the selected FP32 bit pattern, including
        # negativezero and NaNs; no floating-point sum is added to native math.
        value_bits=gl.sum(gl.where(ranks==tile,values.to(gl.int32,bitcast=True),0),0)
        value=value_bits.to(gl.float32,bitcast=True)
        index=gl.sum(gl.where(ranks==tile,indices,0),0)
        take=value>best
        best_index=gl.where(take,index,best_index)
        best=gl.where(take,value,best)
    gl.store(OUT+row*2,best)
    gl.store(OUT+row*2+1,(best_index+OFFSET).to(gl.float32))
    cluster.barrier()
