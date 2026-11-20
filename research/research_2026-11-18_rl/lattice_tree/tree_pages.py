"""Construct the absolute-position SWA page-table suffix for page-one tree verification."""
import triton
import triton.language as tl

@triton.jit
def _tree_suffix(dst, prefix, branches, lens, anc, mapping, row_stride, prefix_stride,
                 branch_stride, window, NODES:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    req=row//NODES
    seq=tl.load(lens+req)
    depth_with_anchor=tl.load(anc+row)
    end=seq+depth_with_anchor
    # Preserve absolute indices. Include the preceding complete physical tile for FA3's tiled window reads.
    first_tile=tl.maximum(0,(end-window-1)//BLOCK-1)
    cols=(first_tile+tl.program_id(1))*BLOCK+tl.arange(0,BLOCK)
    valid=cols<end
    a=tl.load(prefix+req*prefix_stride+cols,mask=valid & (cols<seq),other=0)
    b=tl.load(branches+row*branch_stride+(cols-seq),mask=valid & (cols>=seq),other=0)
    page=tl.where(cols<seq,a,b)
    loc=tl.load(mapping+page.to(tl.int64),mask=valid,other=0).to(tl.int32)
    tl.store(dst+row.to(tl.int64)*row_stride+cols,loc,mask=valid)

def fill_tree_suffix(dst,prefix,branches,lens,anc,mapping,window,nodes):
    assert prefix.stride(1)==branches.stride(1)==dst.stride(1)==mapping.stride(0)==1
    assert branches.shape==(dst.shape[0],nodes) and prefix.shape[0]*nodes==dst.shape[0]
    assert dst.shape[1]>=prefix.shape[1] and window>0
    block=256
    grid=(dst.shape[0],triton.cdiv(window+1,block)+2)
    _tree_suffix[grid](dst,prefix,branches,lens,anc,mapping,dst.stride(0),prefix.stride(0),
                       branches.stride(0),window,nodes,block,num_warps=4)
