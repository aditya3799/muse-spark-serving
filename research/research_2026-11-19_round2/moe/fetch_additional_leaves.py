import concurrent.futures,json,urllib.request,hashlib
from pathlib import Path
here=Path(__file__).resolve().parent/'sources'
items=[]
for p in ['src/ops/linear/nvfp4/nvfp4_codec_unified.cuh','src/ops/linear/nvfp4/nvfp4_a16_mma.cuh','src/ops/linear/nvfp4/nvfp4_a16_sliced_k_mma.cuh','src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu','src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu']:
 items.append(('13thblade/ninfer-all-nvfp4','9b4b8a00760e001191d0177b7f2651b882e77de8',p))
for p in ['kernels/hopper/common/moe_w4a16_grouped_gemm.cu','kernels/hopper/common/w4a16_gemv.cu','kernels/hopper/common/w4a16_gemv_fused.cu','kernels/hopper/common/w4a16_fp8_ldmab.cu']:
 items.append(('Avarok-Cybersecurity/atlas','eae65c07a0daa8073a9edff77199a45254bfa34a',p))
def one(item):
 repo,commit,path=item;url=f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
 try:
  with urllib.request.urlopen(url,timeout=20) as r:data=r.read()
 except Exception as ex:return dict(repository=repo,path=path,error=str(ex),url=url)
 file=repo.split('/')[0]+'__'+path.replace('/','__');(here/file).write_bytes(data)
 return dict(repository=repo,commit=commit,path=path,file=file,bytes=len(data),url=url,sha256=hashlib.sha256(data).hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,items))
(here/'additional_leaf_manifest.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
