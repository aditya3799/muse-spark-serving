"""Derive isolated output-N band materializer from reviewed native full-N source."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;E=P.parent/'expert_partition'
c=(E/'native_materialize.cuh').read_text(encoding='utf-8-sig');w=(E/'native_materialize.py').read_text(encoding='utf-8-sig')
def once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
c=once(c,'#define N_VALUE @N@','#define N_VALUE @N@\n#define SOURCE_N @SOURCE_N@')
c=once(c,'uint32_t quad) {','uint32_t quad, uint32_t row_offset) {')
c=once(c,'const uint32_t nb = within / 1024u;','const uint32_t nb = within / 1024u;\n  const uint32_t source_nb = nb + row_offset / 64u;')
c=once(c,'group * (N_VALUE * 2u) + nb * 128u','group * (SOURCE_N * 2u) + source_nb * 128u')
c=once(c,'group * N_VALUE + nb * 64u','group * SOURCE_N + source_nb * 64u')
c=once(c,'uint2 *__restrict__ output) {','uint2 *__restrict__ output, uint32_t row_offset) {')
assert c.count('write_quad(weight, scales, output, quad)')==2;c=c.replace('write_quad(weight, scales, output, quad)','write_quad(weight, scales, output, quad, row_offset)')
w=once(w,'def render_materializer(n,k,source_transpose,target_transpose,fixed_grid_loop):','def render_materializer(n,k,source_transpose,target_transpose,fixed_grid_loop,source_n):')
w=once(w," text=(P/'native_materialize.cuh').read_text(encoding='utf-8-sig')"," if type(source_n)!=int or source_n<n or source_n%64 or source_n*k>=2**32:raise ValueError('Aligned full source N anduint32capacity required')\n text=(P/'native_materialize_tiled.cuh').read_text(encoding='utf-8-sig')")
w=once(w,"[('@N@',str(n)+'u')","[('@SOURCE_N',str(source_n)+'u'),('@N@',str(n)+'u')") if "[('@N@',str(n)+'u')" in w else w
# Exact current renderer token spelling is recorded; source_N token includes closing@.
w=w.replace("('@SOURCE_N',","('@SOURCE_N@',")
if "('@SOURCE_N@'," not in w:
 token="[('@N@',f'{n}u')";assert token in w;w=w.replace(token,"[('@SOURCE_N@',f'{source_n}u'),('@N@',f'{n}u')")
w=once(w,'class NativeMaterializer:','class TiledMaterializer:')
w=once(w,'def __init__(self,n,k,*,','def __init__(self,source_n,n,k,*,')
w=once(w,'self.n=n;self.k=k;','self.n=n;self.source_n=source_n;self.k=k;')
w=once(w,'render_materializer(n,k,source_transpose,target_transpose,fixed_grid_loop)','render_materializer(n,k,source_transpose,target_transpose,fixed_grid_loop,source_n)')
w=once(w,'def __call__(self,weight,scale,output):','def __call__(self,weight,scale,output,*,row_offset):')
w=once(w,'assert weight.numel()*8==self.n*self.k and scale.numel()*16==self.n*self.k and output.numel()==self.n*self.k',"assert weight.numel()*8==self.source_n*self.k and scale.numel()*16==self.source_n*self.k and output.numel()==self.n*self.k\n  assert type(row_offset)==int and 0<=row_offset<=self.source_n-self.n and row_offset%64==0")
w=once(w,'values=(weight.data_ptr(),scale.data_ptr(),output.data_ptr());types=(ctypes.c_void_p,)*3','values=(weight.data_ptr(),scale.data_ptr(),output.data_ptr(),row_offset);types=(ctypes.c_void_p,)*3+(ctypes.c_uint32,)')
(P/'native_materialize_tiled.cuh').write_text(c);(P/'native_materialize_tiled.py').write_text(w);compile(w,'native_materialize_tiled.py','exec')
(P/'TILED_MATERIALIZER_DERIVATION.json').write_text(json.dumps({'status':'SOURCE_ONLY_UNVALIDATED','changes':['Separate SOURCE_N stride fromtargetN','Runtime64-aligned source rowoffset','Samequarterdecode/scale/bf16uint2store','Unsigneddimensiontokens andbyteintervalaliasguards inherited'],'base_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (E/'native_materialize.cuh',E/'native_materialize.py')},'derived_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'native_materialize_tiled.cuh',P/'native_materialize_tiled.py')}},indent=2)+'\n')
