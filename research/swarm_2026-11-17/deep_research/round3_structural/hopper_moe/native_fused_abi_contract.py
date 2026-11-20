"""Pure-CPU descriptor and18-argument ABI contracts; never calls CUDA or launches."""
import ctypes
from compact_fused_codegen_v2 import admit
DESCRIPTOR_INDICES={0:'A',1:'B',4:'BS'}
ARGUMENT_NAMES=('A','B','C','AS','BS','BZP','Bias','BS2','sorted_ids','expert_ids','num_tokens_padded','expert_layout','tensor_map_buffer','locks','shape_m','top_k','use_int64_expert_layout','post_scales')
def descriptor_plans(c,capacity,layout):
 admit(c)
 if layout!='gate_up_interleaved':raise ValueError('One paired canonical W13 representation required')
 if not c.get('use_warp_spec') or not all(c.get(k,False) for k in ('use_tma_a','use_tma_b','use_tma_bs')) or c.get('use_tma_c'):raise ValueError('Mixed descriptor ABI only for qualified TMA loads/legacyC')
 if not isinstance(capacity,int) or not 0<capacity<=131072:raise ValueError('Invalid fixed route capacity')
 E,N,K=c['num_experts'],c['shape_n'],c['shape_k'];BM,BN,BK=c['block_shape']
 if BK%16 or BN%16 or K%16 or N%16 or BK*16<1024:raise ValueError('Unqualified tensor map tile geometry')
 def plan(dtype,dims,strides,box,swizzle):return {'dtype':dtype,'rank':len(dims),'global_dims':dims,'global_byte_strides':strides,'box_dims':box,'element_strides':[1]*len(dims),'swizzle_bytes':swizzle,'interleave':'NONE','l2_promotion':'NONE','float_oob_fill':'NONE'}
 return {'A':plan('BFLOAT16',[K,capacity],[2*K],[64,BM],128),'B':plan('INT32',[16,N//8,E*K//16],[64,8*N],[16,BN//8,BK//16],0),'BS':plan('UINT8',[16,N//16,E*K//16],[16,N],[16,BN//16,BK//16],0),'tensor_contracts':{'A_shape':[capacity,K],'A_dtype':'bfloat16','B_shape':[E,K//16,2*N],'B_dtype':'int32','BS_shape':[E,K//16,N],'BS_dtype':'float8_e4m3fn','C_shape':[capacity,N//2],'C_dtype':'bfloat16','expert_layout_shape':[E+1],'expert_layout_dtype':'int32','post_scales_shape':[E],'post_scales_dtype':'bfloat16'},'shape_m':capacity,'top_k':1,'use_int64_expert_layout':False,'argument_count':18,'qualification':'Derived from pinned tensor.h/tma.h. Encoding and CUDA launch are untested; this module is CPU-only.'}
class MockArgumentStorage:
 """Own host argument objects; descriptor bytes are deliberately synthetic."""
 def __init__(self,capacity,pointers,descriptor_bytes):
  if set(descriptor_bytes)!=set(DESCRIPTOR_INDICES.values()) or any(len(v)!=128 for v in descriptor_bytes.values()):raise ValueError('Three128-byte encoded descriptors required')
  self.owners=[];self.kinds=[];addresses=[]
  for index,name in enumerate(ARGUMENT_NAMES):
   if index in DESCRIPTOR_INDICES:
    raw=ctypes.create_string_buffer(128+63);address=(ctypes.addressof(raw)+63)&~63;ctypes.memmove(address,descriptor_bytes[name],128);self.owners.append(raw);kind='descriptor_by_value'
   else:
    if index==14:value=ctypes.c_uint32(capacity);kind='uint32'
    elif index==15:value=ctypes.c_uint32(1);kind='uint32'
    elif index==16:value=ctypes.c_bool(False);kind='bool'
    else:value=ctypes.c_void_p(pointers.get(name,0));kind='device_pointer_value'
    self.owners.append(value);address=ctypes.addressof(value)
   addresses.append(address);self.kinds.append(kind)
  self.kernel_params=(ctypes.c_void_p*18)(*addresses)
