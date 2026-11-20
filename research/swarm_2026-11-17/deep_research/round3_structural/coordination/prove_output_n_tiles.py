"""Independent CPU scatter/stride contract, no GPU or serving mutations."""
import json, hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
cases=[];checks=0;negatives=0
for full_n in (4096,8192):
 for tile_n in (128,256,512):
  for rows in (1,17,129,512,1024):
   selected=sorted(set((0,rows//2,rows-1)))
   for row in selected:
    addresses=[]
    for offset in range(0,full_n,tile_n):
     for col in range(tile_n):
      tile_linear=row*tile_n+col
      recovered_row,recovered_col=divmod(tile_linear,tile_n)
      address=recovered_row*full_n+offset+recovered_col
      assert address==row*full_n+offset+col
      addresses.append(address);checks+=1
    assert addresses==list(range(row*full_n,(row+1)*full_n));checks+=1
   if rows>1:
    # Merely offsetting C pointer does not change the pinned legacy writer stride.
    wrong=tile_n;correct=full_n
    assert wrong!=correct;negatives+=1
   k=4096;tiles=full_n//tile_n
   cases.append(dict(full_n=full_n,tile_n=tile_n,rows=rows,tiles=tiles,
    staging_bytes=tile_n*k*2,tile_output_bytes=rows*tile_n*2,
    scatter_logical_read_write_bytes=4*rows*full_n,
    conversion_gemm_scatter_launches=3*tiles))
source=Path('swarm_2026-11-16/moe/humming_vendor/humming/include/humming/epilogue/gmem_writer.cuh')
r=dict(status='CPU_ADDRESS_CONTRACT_ONLY',cases=cases,checks=checks,negative_controls=negatives,
 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 source_findings=['write_legacy row stride hardcoded to ProblemShape::N-PadShape::N',
 'seek also uses that stride for nonindexed row offsets',
 'First prototype must use contiguous tile C then explicit scatter, or independently qualify both stride changes'],
 geometry_caveats=['K remains4096 throughout; no split-K or BF16 partial sums',
 'BN256 baseline can retain BN with Ntile256; Ntile128 requires separate geometry qualification',
 'Ntile256 staging2MiB exceeds nominal local1.5MiBL2; tile1281MiB is only nominal fit, other operands compete'],
 full_helper_launches_without_glue={str(t):3*(8192//t+4096//t) for t in (128,256,512)},
 runtime_validation='Not executed by this proof; full changed-input numerical gate and all-cost timing required')
(P/'OUTPUT_N_TILE_INDEPENDENT_CONTRACT.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),checks=checks,negative_controls=negatives,launches=r['full_helper_launches_without_glue'])))
