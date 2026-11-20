"""Isolated exact candidate-topk communication experiment; never imports SGLang runtime."""
from __future__ import annotations
import argparse, ast, hashlib, json, pathlib, types, typing, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
SOURCE = ROOT / 'sglang/python/sglang/srt/models/dflash.py'
OLD = '''    gathered_vals = tensor_model_parallel_all_gather(vals.float(), dim=-1)
    gathered_ids = tensor_model_parallel_all_gather(global_ids, dim=-1)'''
NEW = '''    if int(lm_head.org_vocab_size) <= (1 << 24):
        # All valid token IDs are exactly representable in FP32. Keep values
        # and IDs in separate leading planes, preserving contiguous top-k rows.
        packed = torch.stack((vals.float(), global_ids.float()), dim=0)
        gathered = tensor_model_parallel_all_gather(packed, dim=-1)
        gathered_vals, gathered_ids = gathered[0], gathered[1]
    else:
        gathered_vals = tensor_model_parallel_all_gather(vals.float(), dim=-1)
        gathered_ids = tensor_model_parallel_all_gather(global_ids, dim=-1)'''

def get_sources():
    source = SOURCE.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'candidate_topk')
    base = ast.get_source_segment(source, node)
    assert base.count(OLD) == 1
    return base, base.replace(OLD, NEW)

def run(device):
    import torch
    torch.set_num_threads(1)
    base, candidate = get_sources()
    rows=[]
    for tp in (1, 2, 4, 8):
      for n in (1, 8, 32, 256):
       for k in (1, 16):
        for pattern in ('random','ties','zeros','extremes'):
         for large_vocab in (False, True):
          generator=torch.Generator(device=device).manual_seed(731+tp+n+k)
          vocab = (1 << 24) + 8 if large_vocab else 202048
          width=257
          tensors=[]
          for rank in range(tp):
           x=torch.randn((n,width),device=device,generator=generator).to(torch.bfloat16)
           if pattern=='ties': x=x.round()
           if pattern=='zeros': x.zero_(); x[:,::2]=-0.0
           if pattern=='extremes': x[:,0]=float('inf'); x[:,1]=float('-inf'); x[:,2]=torch.finfo(torch.bfloat16).max
           tensors.append(x)
          local=[torch.topk(x,k,dim=-1) for x in tensors]
          offsets=[0] if tp==1 else [r*(vocab//tp) for r in range(tp)]
          values=[v.float() for v,i in local]
          ids=[i.long()+offset for (v,i),offset in zip(local,offsets)]
          packed=[torch.stack((v,i.float()),dim=0) for v,i in zip(values,ids)]
          lses=[torch.logsumexp(x.float(),dim=-1) for x in tensors]
          for partition in (False,True):
           outputs=[]; call_counts=[]
           # Every rank is checked for representative small cases; larger cases use rotating ranks.
           ranks=range(tp) if n<=8 else [(n//8)%tp]
           for rank in ranks:
            for version,code in enumerate((base,candidate)):
             calls=[]
             def gather(value,dim=-1):
              if value.ndim==3:
               peers=packed; kind='packed'
              elif value.shape[-1]==1 and partition and len(calls)>=(1 if version and not large_vocab else 2):
               peers=[v.unsqueeze(-1) for v in lses];kind='lse'
              elif value.dtype==torch.int64:
               peers=ids;kind='ids'
              else:
               peers=values;kind='vals'
              assert torch.equal(value,peers[rank]),(kind,value.shape)
              calls.append(kind)
              return torch.cat(peers,dim=dim)
             head=types.SimpleNamespace(weight=torch.empty(1,device=device),quant_method=None,org_vocab_size=vocab,shard_indices=types.SimpleNamespace(num_org_elements=width,org_vocab_start_index=offsets[rank]))
             ns={'torch':torch,'nn':torch.nn,'Tuple':typing.Tuple,'Optional':typing.Optional,'get_parallel':lambda:types.SimpleNamespace(tp_size=tp),'should_apply_lm_head_quant_method':lambda *a:False,'is_dense_head_weight':lambda x:True,'_project_candidate_logits':lambda *a,**kw:tensors[rank],'_radix_topk':lambda x,k:torch.topk(x,k,dim=-1),'lilicorr_topk_lse':lambda x,k:(*torch.topk(x,k,dim=-1),torch.logsumexp(x.float(),dim=-1)),'tensor_model_parallel_all_gather':gather}
             exec(compile(code,str(SOURCE),'exec'),ns)
             result=ns['candidate_topk'](torch.empty((n,1),device=device),head,k,with_partition=partition)
             outputs.append(result);call_counts.append(len(calls))
             if version:
              for a,b in zip(outputs[-2],result):
               assert (a is None and b is None) or (a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.view(torch.uint8),b.view(torch.uint8)))
              expected=0 if tp==1 else 2+int(partition)
              assert call_counts[-2]==expected
              assert call_counts[-1]==expected-int(tp>1 and not large_vocab)
           rows.append({'tp':tp,'rows':n,'k':k,'pattern':pattern,'large_vocab_fallback':large_vocab,'with_partition':partition,'rank_checks':len(list(ranks)),'bitwise':True,'collective_counts':[call_counts[-2],call_counts[-1]]})
    report={'status':'CPU_SOURCE_EXTRACTED_CORRECTNESS' if device=='cpu' else 'GPU_EMULATED_COLLECTIVE_CORRECTNESS','device':device,'torch':torch.__version__,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'base_function_sha256':hashlib.sha256(base.encode()).hexdigest(),'candidate_function_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'cases':len(rows),'rank_cases':sum(r['rank_checks'] for r in rows),'all_bitwise':True,'rows':rows,'limits':['All-gather emulated by tensor concatenation; no NCCL or distributed capture validation','No serving performance result','NaN logits intentionally not in supported serving fixtures; signed zero, infinities and finite ties included','Actual FlashInfer local top-k unchanged but fixtures use torch.topk','Large-vocabulary branch retains original two-gather path']}
    (OUT/('SELECTOR_PACK_'+device.upper()+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'selector_candidate_function.py').write_text(candidate+'\n')
    print(json.dumps({k:report[k] for k in ('status','cases','rank_cases','all_bitwise','source_sha256')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');args=parser.parse_args();run(args.device)
