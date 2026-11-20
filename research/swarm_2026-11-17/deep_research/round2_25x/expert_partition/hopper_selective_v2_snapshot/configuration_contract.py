"""Metadata-only exact M2 selection; independent of request token count."""
def m2_compatible(heavy,m2,cutoff,sm_version=None):
 if sm_version!=86:return False
 expected={'c_dtype':'bfloat16','weight_scale_group_size':16,'weight_scale_type':'group','weight_scale_2_type':'none','num_stages':3,'num_write_splits':1,'pad_shape_n':0,'pad_shape_k':0,'use_cp_async':True,'num_experts':32,'shape_n':8192,'shape_k':4096,'gemm_type':'indexed'}
 if any(heavy.get(k)!=v or m2.get(k)!=v for k,v in expected.items()):return False
 if any(v for c in (heavy,m2) for k,v in c.items() if k.startswith('use_tma')):return False
 if heavy.get('input_scale_group_size',0) or heavy.get('use_fused_e8m0_scale',False) or heavy.get('use_int_weight_scale',False):return False
 if cutoff!=16:return False
 if heavy.get('block_shape')!=[16,256,128] or heavy.get('warp_shape')!=[16,64,64]:return False
 if m2.get('block_shape')!=[32,256,128] or m2.get('warp_shape')!=heavy['warp_shape']:return False
 if heavy.get('mma_type')!='mma' or heavy.get('a_dtype')!='bfloat16' or heavy.get('b_dtype')!='float4e2m1' or heavy.get('bs_dtype')!='float8e4m3':return False
 if any(heavy.get(k,False) for k in ('use_stream_k','use_f16_accum','use_warp_spec','use_packed_k_layout','has_bias','has_zero_point')):return False
 lhs={k:v for k,v in heavy.items() if k not in ('block_shape','num_sms')};rhs={k:v for k,v in m2.items() if k not in ('block_shape','num_sms')}
 return lhs==rhs
