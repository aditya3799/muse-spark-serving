"""CPU Humming A16/NVFP4 per-expert slot adapter, pinned aligned layout only."""
import ast,pathlib,types,numpy as np
R=next(p for p in pathlib.Path(__file__).resolve().parents if (p/'RULES.md').exists())
V=R/'swarm_2026-11-16/moe/humming_vendor/humming'
PORT=R/'swarm_2026-11-17/deep_research/round3_structural/aux_speculation/test_packed_addresses.py'
ns={};tree=ast.parse(PORT.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(PORT),'exec'),ns)

class ByteTensor:
    def __init__(self,a):self.a=np.asarray(a)
    @property
    def shape(self):return self.a.shape
    @property
    def device(self):return 'cpu'
    def transpose(self,a,b):return ByteTensor(self.a.swapaxes(a,b))
    def contiguous(self):return ByteTensor(np.ascontiguousarray(self.a))
    def view(self,*shape):
        if len(shape)==1 and isinstance(shape[0],tuple):shape=shape[0]
        return ByteTensor(self.a.reshape(shape))
    def __getitem__(self,i):return ByteTensor(self.a[i])

fake=types.SimpleNamespace(Tensor=ByteTensor,int32=np.int32,tensor=lambda v,**kw:np.asarray(v,dtype=kw['dtype']))
tree=ast.parse((V/'transform.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='transform_humming_weight_scale')
g={'torch':fake};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(V/'transform.py'),'exec'),g)

def pack_codes(codes,transpose):
    e,n,k=codes.shape
    if codes.dtype!=np.uint8 or np.any(codes>15) or n%64 or k%64:raise ValueError('aligned FP4 nibble source')
    ref=ns['reference'](n,k,4,transpose)
    out=np.zeros((e,k//16,n*2),np.uint32)
    for pos,(row,col) in ref.items():out.reshape(e,-1)[:,pos//8]|=codes[:,row,col].astype(np.uint32)<<np.uint32(4*(pos%8))
    return out

def unpack_codes(packed,n,k,transpose):
    if packed.dtype!=np.uint32 or packed.shape[1:]!=(k//16,n*2):raise ValueError('packed shape')
    out=np.empty((len(packed),n,k),np.uint8);flat=packed.reshape(len(packed),-1)
    for row in range(n):
        for col in range(k):
            pos=ns['direct'](n,k,row,col,4,transpose)
            out[:,row,col]=((flat[:,pos//8]>>np.uint32(4*(pos%8)))&15).astype(np.uint8)
    return out

def transform_scales(raw):
    if raw.dtype!=np.uint8 or raw.ndim!=3 or raw.shape[1]%64:raise ValueError('FP8 raw-byte scale shape')
    return g['transform_humming_weight_scale'](ByteTensor(raw)).a

class Bank:
    def __init__(self,capacity,h,i,transpose):
        if min(capacity,h,i)<=0 or h%64 or i%64:raise ValueError('capacity/alignment')
        self.capacity,self.h,self.i,self.transpose=capacity,h,i,transpose
        self.tensors={}
        for stage,n,k in [('w13',2*i,h),('w2',h,i)]:
            self.tensors[stage+'_weight']=np.zeros((capacity,k//16,n*2),np.uint32)
            self.tensors[stage+'_weight_scale']=np.zeros((capacity,k//16,n),np.uint8)
            self.tensors[stage+'_post_scale']=np.zeros(capacity,np.uint16) # BF16 bits, no recast
    def install(self,stage,codes,scales,post_bits):
        n,k=codes.shape[1:]
        w=pack_codes(codes,self.transpose);sf=transform_scales(scales)
        if w.shape!=self.tensors[stage+'_weight'].shape or sf.shape!=self.tensors[stage+'_weight_scale'].shape:raise ValueError('stage dimensions')
        np.copyto(self.tensors[stage+'_weight'],w);np.copyto(self.tensors[stage+'_weight_scale'],sf);np.copyto(self.tensors[stage+'_post_scale'],post_bits)
    def copy_slot(self,dst,other,src):
        if (self.h,self.i,self.transpose)!=(other.h,other.i,other.transpose):raise ValueError('architecture/layout/shape mismatch')
        if not 0<=dst<self.capacity or not 0<=src<other.capacity:raise ValueError('slot bounds')
        for key,target in self.tensors.items():target[dst]=other.tensors[key][src]
    def bytes_per_slot(self):return sum(v[0].nbytes if v.ndim>1 else v.dtype.itemsize for v in self.tensors.values())
