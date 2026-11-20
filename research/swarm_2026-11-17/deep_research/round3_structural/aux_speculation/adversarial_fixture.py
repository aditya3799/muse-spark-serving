"""Deterministic cancellation/exponent fixtures; CPU preparation only until leased."""
import torch

def weight_fixture(n,k,stage,expert=0,generation=0,device='cpu'):
    assert n%64==0 and k%64==0
    # Half-K anti-pairs have identical positive group scales and opposite codes.
    row=torch.arange(n,device=device,dtype=torch.int32)[:,None]
    col=torch.arange(k//4,device=device,dtype=torch.int32)[None,:]
    low=(row*3+col*5+expert*7+generation*3)%16
    high=(row*7+col*3+expert*5+generation)%16
    left=(low|(high<<4)).to(torch.uint8)
    right=left.bitwise_xor(0x88)
    # One residual group keeps every row from becoming a vacuous exact zero.
    right[:,-8:]=left[:,-8:]
    weight=torch.cat((left,right),dim=1).contiguous()
    group=torch.arange(k//32,device=device,dtype=torch.int32)[None,:]
    sb=((row*13+group*17+expert*19+generation*23)%127).to(torch.uint8)
    scales=torch.cat((sb,sb),dim=1).contiguous().view(torch.float8_e4m3fn)
    base=2.**(-16 if stage=='w13' else -12)
    global_scale=torch.tensor(base*(1.,1.5,.75)[generation%3]*(1+expert/8),device=device,dtype=torch.bfloat16)
    return weight,scales,global_scale

def activation_fixture(rows,k,generation=0,device='cpu'):
    assert k%64==0
    row=torch.arange(rows,device=device,dtype=torch.int32)[:,None]
    col=torch.arange(k//2,device=device,dtype=torch.int32)[None,:]
    exponent=((col*7+row*3+generation*5)%33)-28
    mantissa=1.+((col*11+row*17)%128).float()/128
    sign=torch.where((col+row)%2==0,1.,-1.)
    left=(torch.ldexp(mantissa,exponent)*sign).to(torch.bfloat16)
    right=left.clone()
    # Adjacent BF16 mantissas at selected coordinates leave small cancellation residuals.
    perturbed=(left.float()*(1.+1./128)).to(torch.bfloat16)
    right=torch.where((col+generation)%17==0,perturbed,right)
    return torch.cat((left,right),dim=1).contiguous()

def cpu_contract():
    import json,hashlib
    from pathlib import Path
    checks=0;records=[]
    for generation in range(3):
      for expert in range(2):
       for stage,n in (('w13',128),('w2',64)):
        w,s,g=weight_fixture(n,4096,stage,expert,generation)
        assert w.shape==(n,2048) and s.shape==(n,256)
        assert torch.isfinite(s.float()).all() and torch.isfinite(g)
        assert torch.equal(s.view(torch.uint8)[:,:128],s.view(torch.uint8)[:,128:])
        assert torch.equal(w[:,:1016].bitwise_xor(0x88),w[:,1024:2040])
        unique=sorted(s.view(torch.uint8).unique().tolist());assert unique==list(range(127))
        x=activation_fixture(7,4096,generation);assert torch.isfinite(x).all()
        assert x.abs().max()<33 and x[x!=0].abs().min()<1e-7
        assert not torch.equal(x[:,:2048],x[:,2048:])
        checks+=8
        records.append({'generation':generation,'expert':expert,'stage':stage,'positive_scale_bytes':len(unique),'global_scale':float(g),'x_min_nonzero':float(x[x!=0].abs().min()),'x_max_abs':float(x.abs().max())})
    p=Path(__file__).resolve();result={'status':'CPU_FIXTURE_CONTRACT_PASS_NOT_GPU_VALIDATED','checks':checks,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Full positive finite E4M3 scale encoding range0..126; all16 FP4 codes; BF16 input spread and half-K anti-pairs, nonzero residual group, dynamic weights/scales/globals. Negative FP8 scales are outside physical positive quantization-scale fixture, already included in operand enumeration.','cases':records}
    p.with_name('ADVERSARIAL_CPU_CONTRACT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'checks':checks}))
if __name__=='__main__':cpu_contract()
