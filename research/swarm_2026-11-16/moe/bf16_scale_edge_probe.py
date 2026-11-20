"""Exhaustive finite BF16 input multiplication edge proof for masked scale."""
from pathlib import Path
import argparse,hashlib,json,sys
import torch
from humming_scaled_rows import scale_active_rows
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
values=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
values=values[torch.isfinite(values)].cuda()
scale_values=[0.,-0.,1.,-1.,.1,-.1,.5,1.5,2.**-133,2.**-126,2.**-7,2.**7,2.**126,-2.**126]
scales=torch.tensor(scale_values,device='cuda',dtype=torch.bfloat16)
source=values.expand(len(scales),-1).contiguous();out=source.clone()
ids=torch.arange(len(scales),device='cuda',dtype=torch.int32).view(-1,1)
scale_active_rows(out,ids,scales);torch.cuda.synchronize()
reference=source*scales[:,None]
assert torch.equal(out.view(torch.int16),reference.view(torch.int16)), 'BF16 product bit mismatch'
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3):scale_active_rows(out,ids,scales)
torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):scale_active_rows(out,ids,scales)
rows=[]
for mode in ('reversed','invalid','mixed','identity'):
    out.copy_(source)
    new_ids=torch.arange(len(scales),device='cuda',dtype=torch.int32)
    if mode=='reversed':new_ids=new_ids.flip(0)
    if mode=='invalid':new_ids.fill_(-1)
    if mode=='mixed':new_ids[::2]=-1
    ids.copy_(new_ids[:,None]);graph.replay();torch.cuda.synchronize()
    selected=scales[new_ids.clamp_min(0).long()]
    expected=source*selected[:,None]
    expected=torch.where((new_ids>=0)[:,None],expected,source)
    equal=torch.equal(out.view(torch.int16),expected.view(torch.int16));assert equal, mode
    rows.append({'mode':mode,'bitwise_equal':equal})
report={'passed':True,'finite_bf16_patterns':values.numel(),'scale_values_after_bf16':[float(x) for x in scales.cpu()],'gpu':torch.cuda.get_device_name(),'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'kernel_sha256':hashlib.sha256((Path(__file__).parent/'humming_scaled_rows.py').read_bytes()).hexdigest(),'eager_bitwise_equal':True,'graph_rows':rows,'scope':'Finite BF16 bit-pattern inputs, including signed zero/subnormals/large finite values; outputs may overflow to infinities. No NaN-payload preservation claim.'}
(a.out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);torch.cuda.synchronize()
