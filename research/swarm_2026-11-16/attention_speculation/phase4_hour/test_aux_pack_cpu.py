"""Real PyTorch CPU bit-pattern/lifecycle proof; no GPU calls or model run."""
import hashlib,json
from pathlib import Path
import torch
from aux_pack_contract import make_aux_owner
P=Path(__file__).resolve().parent
torch.manual_seed(6216)
ids=(1,16,30,44,59);h=64;maxn=512
owner=make_aux_owner(torch,max_tokens=maxn,hidden=h,capture_ids=ids,num_layers=62,device='cpu')
assert owner is not None
address=owner.buffer.data_ptr();checks=0;negative=[]
for n in (512,256,128,64,32,8,3,511,17,512):
    capture=owner.begin(n);reference=[]
    for layer in ids:
        bits=torch.randint(-(1<<31),(1<<31)-1,(n,h),dtype=torch.int32)
        # Includes signed zeros, subnormals, infinities and NaN payloads.
        special=torch.tensor([0,-2147483648,1,8388607,2139095040,-8388608,2143289345,2147483647],dtype=torch.int32)
        bits.view(-1)[:8]=special
        stream=bits.view(torch.float32)
        reference.append(stream.to(torch.bfloat16))
        capture.capture(layer,stream)
        stream.zero_() # borrowed/reused stream must not alter captured tap.
    out=capture.finish();expected=torch.cat(reference,dim=-1)
    assert torch.equal(out.view(torch.int16),expected.view(torch.int16)),'Bitwise FP32->BF16 pack mismatch'
    assert out.data_ptr()==address
    adopted=owner.adopt_bcg_output(out,maxn)
    assert adopted is not None and adopted.data_ptr()==address and adopted.shape==(maxn,h*5)
    assert owner.adopt_bcg_output(out.clone(),maxn) is None
    assert owner.adopt_bcg_output(out,n-1) is None
    assert owner.adopt_bcg_output(owner.buffer[1:],maxn) is None
    try:owner.begin(n)
    except RuntimeError:negative.append('overwrite_before_materialization')
    else:raise AssertionError('Aux overwrite')
    owner.release_after_materialization(capture.generation);checks+=1
    try:capture.finish()
    except RuntimeError:negative.append('finish_after_release')
    else:raise AssertionError('Stale finish')
    newer=owner.begin(n)
    try:capture.finish()
    except RuntimeError:negative.append('finish_after_new_generation')
    else:raise AssertionError('Old lease returned new output')
    owner.release_after_materialization(newer.generation)
    try:owner.release_after_materialization(capture.generation)
    except RuntimeError:negative.append('stale_release')
    else:raise AssertionError('Stale release')
for kwargs in ({'sp':True},{'dp':2},{'pp':2},{'phase':'decode'},{'capture_ids':(1,1)},{'capture_ids':(62,)},{'capture_ids':(16,1)},{'capture_ids':()},{'capture_ids':(1,'2')},{'capture_ids':None},{'max_tokens':'512'},{'hidden':True}):
    params=dict(max_tokens=maxn,hidden=h,capture_ids=ids,num_layers=62,device='cpu');params.update(kwargs)
    assert make_aux_owner(torch,**params) is None;checks+=1
bad=owner.begin(8)
try:bad.capture(16,torch.zeros((8,h)))
except ValueError:negative.append('wrong_capture_order')
else:raise AssertionError('order')
try:bad.finish()
except RuntimeError:negative.append('missing_taps')
else:raise AssertionError('missing taps')
owner.release_after_materialization(bad.generation)
report={'status':'PASS','checks':checks,'direct_cast_bitwise_cases':10,'negative_guards':negative,'stable_output_pointer':True,
        'scope':'Actual PyTorch CPU conversion incl NaN payloads plus ownership. CUDA casting/graph replay/model integration and async consumer ordering remain unproven.',
        'torch':torch.__version__,'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'aux_pack_contract.py',Path(__file__))}}
(P/'cpu_aux_pack_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
