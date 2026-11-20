"""Four ID/score dtype contracts, interleaved in the same timing blocks."""
import hashlib,json,random,traceback,uuid,zipfile
from datetime import datetime,timezone
from pathlib import Path
import torch
if 'UpdateStudy' not in globals():
    from update_study import UpdateStudy
    from source_slices import BaselineSourceSlice,SharedSourceSlice
    from timing_support import routes,expert_mapping,write_csv,parse_trace

CONTRACTS={'i64_s16':('int64','float16'),'i32_s16':('int32','float16'),
           'i64_s32':('int64','float32'),'i32_s32':('int32','float32')}

class ContractStudy(UpdateStudy):
    def __init__(self,manifest,helper_source,output_root=None,force_cpu=False,samples=11,repeats=3,trials=3):
        self.manifest,self.helper_source=manifest,helper_source
        self.t,self.gpu,self.force_cpu=None,False,force_cpu
        self.samples,self.repeats,self.trials=samples,repeats,trials
        root=Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        self.out=root/('v4_router_contract_'+run_id);self.out.mkdir(parents=True,exist_ok=False)
        self.rows,self.raw,self.cases=[],[],[]
        self.report={'schema_version':10,'run_id':run_id,'source_manifest':manifest,
            'scope':'Source-prefix combined graphs with four ID/score dtype combinations. Synthetic routing inputs matching default output dtypes, not execution of the router or complete model.',
            'config':{'tokens':[9,32,128],'stages':[1,8],'contracts':CONTRACTS,'trials':trials,'samples':samples,'repeats':repeats},
            'contract':'Maps fixed after setup. Inputs contiguous during timing. Hidden activations remain synthetic float16 tensors of width 32; this prefix does not compute on their values. Same .125 scores in all formats.',
            'sections':{}}
        self.save()

    def contract_inputs(self,n,contract,strided=False):
        ids_dtype,scores_dtype=CONTRACTS[contract]
        inputs=list(self.inputs(n,ids_dtype,strided=strided))
        inputs[1]=torch.full((n,8),.125,dtype=getattr(torch,scores_dtype),device=inputs[2].device)
        return tuple(inputs)

    def input_contract(self):
        def run():
            count=0;outputs=0;fused=0;aliases=0
            with torch.inference_mode():
                for contract in CONTRACTS:
                    for n in (0,1,8,9,32):
                        for strided in (False,True):
                            modules=self.modules();selected={k:modules[k] for k in ('baseline','single_owner')}
                            inputs=self.contract_inputs(n,contract,strided);values=routes(n,'sentinel');mapping=expert_mapping('contiguous')
                            inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(n,8))
                            for m in selected.values():
                                output=m(*inputs);fused+=int(self.check_result(output,m,inputs,values,mapping));outputs+=1
                                if n:
                                    same=output[2].data_ptr()==inputs[1].data_ptr()
                                    assert same==(inputs[1].dtype==torch.float32)
                                    aliases+=int(same)
                            count+=1
            return {'status':'passed','input_cases':count,'output_checks':outputs,'deferred_fused_outputs':fused,'float32_score_alias_checks':aliases,
                'note':'Fused outputs checked only for branch/metadata, never uninitialized ID values. Noncontiguous ID inputs are correctness controls only. Actual router not called.'}
        return self.section('input_contract',run)

    def ready(self):return self.gpu and all(self.report['sections'].get(s,{}).get('status')=='passed' for s in ('lifecycle','input_contract'))

    def make_contract_case(self,n,depth):
        states={};variants={}
        mappings=[expert_mapping(('contiguous','shared','permuted')[i%3]) for i in range(depth)]
        def replay(capture):
            graph,output,_=capture
            def fn():graph.replay();return output
            return fn
        for contract in CONTRACTS:
            inputs=self.contract_inputs(n,contract)
            inputs[2].copy_(torch.tensor(routes(n,'hot_local'),dtype=inputs[2].dtype).reshape(n,8))
            for name,cls in [('baseline',BaselineSourceSlice),('single',SharedSourceSlice)]:
                with torch.inference_mode(False):chain=[cls().cuda() for _ in range(depth)]
                for m,mapping in zip(chain,mappings):m._expert_map.copy_(torch.tensor(mapping,dtype=torch.int32,device='cuda'))
                capture=self.capture(lambda chain=chain,inputs=inputs:tuple(m(*inputs) for m in chain))
                key=name+'_'+contract;states[key]=(chain,inputs,mappings,capture);variants[key]=replay(capture)
        return states,variants

    def check_contract_case(self,bundle,n):
        states,variants=bundle;checks=0
        for pattern in ('sentinel','hot_local'):
            values=routes(n,pattern)
            for name,fn in variants.items():
                chain,inputs,mappings,_=states[name]
                inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(n,8))
                result=fn();assert len(result)==len(chain)
                for output,m,mapping in zip(result,chain,mappings):
                    assert not self.check_result(output,m,inputs,values,mapping)
                    assert (output[2].data_ptr()==inputs[1].data_ptr())==(inputs[1].dtype==torch.float32)
                    checks+=1
        return checks,values

    def benchmark(self):
        def run():
            if not self.ready():return {'status':'skipped','reason':'Requires passing CUDA lifecycle and input-contract gates'}
            with torch.inference_mode():
                for trial in range(self.trials):
                    configs=[(n,d) for n in (9,32,128) for d in (1,8)];random.Random(10200+trial).shuffle(configs)
                    for index,(n,depth) in enumerate(configs):
                        identity={'trial':trial,'case_order':index,'tokens':n,'pattern':'hot_local','scope':'router_contract','stages':depth};bundle=None
                        try:
                            bundle=self.make_contract_case(n,depth);checks,values=self.check_contract_case(bundle,n)
                            self.measure_updates(bundle[1],identity)
                            for name,fn in bundle[1].items():
                                chain,inputs,mappings,_=bundle[0][name]
                                for output,m,mapping in zip(fn(),chain,mappings):self.check_result(output,m,inputs,values,mapping)
                            self.cases.append(dict(identity,status='passed',changed_input_output_checks=checks))
                        except Exception as exc:
                            self.rows=[r for r in self.rows if not all(r[k]==v for k,v in identity.items())]
                            self.raw=[r for r in self.raw if not all(r[k]==v for k,v in identity.items())]
                            self.cases.append(dict(identity,status='failed',error=repr(exc),traceback=traceback.format_exc(limit=6)))
                        finally:torch.cuda.synchronize();bundle=None
                        if (index+1)%3==0:self.save();print('Router-contract trial',trial+1,index+1,'/',len(configs),flush=True)
            failed=sum(c['status']!='passed' for c in self.cases)
            return {'status':'passed' if not failed else 'partial','cases':len(self.cases),'failed_cases':failed,'changed_input_output_checks':sum(c.get('changed_input_output_checks',0) for c in self.cases)}
        return self.section('benchmark',run)

    def profile(self):
        def run():
            if not self.ready():return {'status':'skipped','reason':'Requires passing CUDA gates'}
            with torch.inference_mode():
                bundle=self.make_contract_case(128,8);self.check_contract_case(bundle,128);variants=bundle[1]
                for fn in variants.values():
                    for _ in range(5):fn()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as prof:
                    for i in range(3):
                        names=list(variants);random.Random(10300+i).shuffle(names)
                        for name in names:
                            with torch.profiler.record_function('mapping/'+name):variants[name]()
                        prof.step()
                    torch.cuda.synchronize()
                path=self.out/'router_contract_trace.json';prof.export_chrome_trace(str(path));result=parse_trace(path,self.out)
                result['note']='Separate combined-graph trace: eight stages, 128 tokens, three cycles per dtype/implementation variant. Missing graph correlation or CPU-only trace yields partial status. Count actual device operations, not annotations.'
                return result
        return self.section('profiler',run)

    def export(self):
        comparisons=[]
        pairs=[('single_'+c,'baseline_'+c,'mapping_candidate') for c in CONTRACTS]
        controls=[('i32_s32','i64_s16'),('i32_s16','i64_s16'),('i64_s32','i64_s16'),('i32_s32','i32_s16'),('i32_s32','i64_s32')]
        pairs += [(v+'_'+c,v+'_'+b,'dtype_control') for v in ('baseline','single') for c,b in controls]
        for case in self.cases:
            if case['status']!='passed':continue
            key={k:case[k] for k in ('trial','tokens','pattern','scope','stages')}
            rows={r['variant']:r for r in self.rows if all(r[k]==v for k,v in key.items())}
            for candidate,baseline,kind in pairs:
                row=dict(key,candidate=candidate,baseline=baseline,comparison_kind=kind)
                for metric in ('event','wall','enqueue'):
                    b,c=rows[baseline][metric+'_us_median'],rows[candidate][metric+'_us_median']
                    row.update({metric+'_baseline_us':b,metric+'_candidate_us':c,metric+'_reduction_percent':100*(1-c/b),metric+'_baseline_us_per_stage':b/case['stages'],metric+'_candidate_us_per_stage':c/case['stages']})
                comparisons.append(row)
        self.report['totals']={'cases':len(self.cases),'timing_rows':len(self.rows),'raw_samples':len(self.raw),'comparisons':len(comparisons)}
        self.save();write_csv(self.out/'comparisons.csv',comparisons)
        lines=['NOTEBOOK 10: ROUTER OUTPUT DTYPE CONTRACT','Run: '+self.report['run_id'],'',self.report['scope'],'']
        for name in ('environment','lifecycle','input_contract','benchmark','profiler'):lines.append(name+': '+self.report['sections'].get(name,{}).get('status','not_run'))
        lines+=['','i64_s16 is the historical harness. i32_s32 matches inspected default JIT/Triton router output dtypes. Mixed variants isolate one conversion at a time.',
            'All eight variants are interleaved within every sample. Complete chains are measured; per-stage values divide chain time. The four contracts are synthetic inputs; actual router kernels and runtime output buffers are not executed.',
            'Maps fixed, IDs and scores contiguous during timing. Hidden activations remain float16 width-32 placeholders; the prefix only checks their shape/contiguity/device.',
            'CUDA-event, wall and host enqueue times overlap. Never add/subtract them to infer exact CPU/GPU shares. Dtype-control reductions quantify harness overhead, not a new serving optimization.',
            'Source and drafts unchanged. No actual alignment/GEMMs, attention, communication, small fused kernel, model initialization or tokenizer/scheduler startup. No H200/full-serving claim.','']
        (self.out/'SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name!='checksums.json'}
        (self.out/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        archive=self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():z.write(p,p.name)
        print('\n'.join(lines));print('Download:',archive);return archive
