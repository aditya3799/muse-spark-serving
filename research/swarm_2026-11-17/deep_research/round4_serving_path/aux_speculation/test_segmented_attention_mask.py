import hashlib,itertools,json
from pathlib import Path
P=Path(__file__).parent;ROOT=P.parents[3]
wrapper=ROOT/'sglang/python/sglang/kernels/ops/attention/flash_attention.py'
backend=ROOT/'sglang/python/sglang/srt/layers/attention/flashattention_backend.py'
assert 'causal mask is aligned to the bottom right' in wrapper.read_text()
assert 'i + seqlen_k - seqlen_q - window_size[0]' in wrapper.read_text()
def interval(nq,nk,i,left):
    absolute=i+nk-nq
    return (0 if left<0 else max(0,absolute-left),min(nk-1,absolute))
checks=cases=0
for prefix,cut,left in itertools.product((0,1,7,127,2047,2048,4097,131064),(1,2,4,6,7),(-1,0,15,2047)):
    for i in range(8):
        baseline=interval(8,prefix+8,i,left)
        actual=interval(cut,prefix+cut,i,left) if i<cut else interval(8-cut,prefix+8,i-cut,left)
        assert actual==baseline;checks+=1
    cases+=1
assert interval(4,40+4,0,-1)!=interval(4,40+8,0,-1)
r={'status':'CPU_SOURCE_MASK_AND_CACHE_BOUNDARY_PROOF','cases':cases,'query_interval_equalities':checks,
 'formula':'FA wrapper documents bottom-right causal alignment; query absolute position=i+cachelen-querylen. Stage1 uses(L+cut,cut), stage2(L+8,8-cut), exactlymatching original(L+8,8).',
 'negative_control':'Tail retaining old prefixL givesincorrect absolutequerypositionL instead ofL+4',
 'cache_precision_source':'Standard nonMLA,nonCP,nonembedding FA3 forward_extend writes currentK/V to pool before attention; flash_attn_with_kvcache receives stored key_cache/value_cache without rawk/v arguments. Thus firstsegmenttokens already had cachequantization when attended within originalverify8; segmentation doesnot itself add a new cachequantization boundary.',
 'scope_guards':['Generation mode sofa_skip_kv_cache=False','MHA notMLA','topk1 no treecascade','Causal full/SWA','NoCP/rawKVembeddingpath'],
 'remaining_arithmetic_risks':['ProjectedK/V values maychange withtarget GEMMrowshape','FA num_splits0 mayselectdifferent SplitKV sizing; equal masks do notproveequal floatingpoint reductions','All model layers/MoE/normalization dispatch and greedy outputs needrealmodelproof'],
 'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (wrapper,backend)},'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(P/'SEGMENTED_ATTENTION_MASK_CPU.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))