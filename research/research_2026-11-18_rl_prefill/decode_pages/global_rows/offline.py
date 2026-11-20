"""Compile both scored-capacity metadata variants on the offline SM90a target."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
H=Path(__file__).resolve().parent
records=[]
for name,path in [('parent',H.parent/'parent_metadata.py'),('suffix',H.parent/'metadata.py'),('global',H/'metadata.py')]:
    spec=importlib.util.spec_from_file_location('fa3_'+name,path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    f=m._fused_decode_metadata_kernel
    for query in [1,8,13]:
        constants={'PAGE_SIZE':1,'SEQ_LEN_DELTA':query if query>1 else 0,'BLOCK_COLS':128 if name=='parent' else 256,'SKIP_PAGE_TABLE':False,'ZERO_TAIL':False,'full_v2p':None,'swa_v2p':None}
        if name!='parent':constants['SWA_SUFFIX_TOKENS']=2047+query
        pointers=dict(seq_lens='*i32',req_to_token='*i32',req_pool_indices='*i64',cache_seqlens_int32='*i32',cu_seqlens_k='*i32',page_table='*i32',swa_page_table='*i32',full_to_swa_index_mapping='*i64')
        if name=='global':pointers['DIRECT_FULL_INDICES']='*i32'
        sig={key:('constexpr' if key in constants else pointers.get(key,'i32')) for key in f.arg_names}
        out=triton.compile(ASTSource(f,signature=sig,constexprs=constants),target=GPUTarget('cuda',90,32),options={'num_warps':constants['BLOCK_COLS']//32,'num_stages':3})
        raw=out.asm['cubin'];records.append(dict(variant=name,q=query,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),target=next(s for s in out.asm['ptx'].splitlines() if s.startswith('.target'))))
print(json.dumps(records));(H/'offline_sm90.json').write_text(json.dumps(records,indent=2))

