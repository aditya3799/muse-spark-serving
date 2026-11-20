"""Compare existing cache ownership drafts under a static native-dtype workload."""
import hashlib,json,random,traceback,uuid,zipfile
from datetime import datetime,timezone
from pathlib import Path
import torch
if 'ContractStudy' not in globals():
    from contract_study import ContractStudy
    from source_slices import BaselineSourceSlice,CandidateSourceSlice,SharedSourceSlice
    from timing_support import routes,expert_mapping,write_csv,parse_trace

class CacheStudy(ContractStudy):
    def __init__(self,manifest,helper_source,output_root=None,force_cpu=False,samples=11,repeats=3,trials=3):
        self.manifest,self.helper_source=manifest,helper_source
        self.t,self.gpu,self.force_cpu=None,False,force_cpu
        self.samples,self.repeats,self.trials=samples,repeats,trials
        root=Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        self.out=root/('v4_cache_choice_'+run_id);self.out.mkdir(parents=True,exist_ok=False)
        self.rows,self.raw,self.cases=[],[],[]
        self.report={'schema_version':11,'run_id':run_id,'source_manifest':manifest,
            'scope':'Compare unchanged baseline, two-buffer and single-owner source prefixes with static maps and native router-default dtypes. No full serving integration.',
            'config':{'tokens':[9,32,128,512],'stages':[1,8],'patterns':['hot_local','uniform'],'trials':trials,'samples':samples,'repeats':repeats,'ids_dtype':'int32','scores_dtype':'float32','schedules':['eager','graph']},
            'contract':'Each stage owns a distinct module/map, with local experts 64..95 of 256. Maps stay fixed during timing. Stages are independent; hidden activations are float16 width-32 placeholders.',
            'sections':{}}
        self.save()

    def preservation_contract(self):
        def run():
            with torch.inference_mode():
                modules=self.modules();b,t,s=modules['baseline'],modules['two_buffer'],modules['single_owner']
                assert set(dict(b.named_buffers()))=={'_expert_map'}
                assert set(dict(t.named_buffers()))=={'_expert_map','_expert_lookup'}
                assert set(dict(s.named_buffers()))=={'_expert_lookup'}
                assert b._expert_map.storage_offset()==t._expert_map.storage_offset()==0
                assert s._expert_map.storage_offset()==1
                assert t._expert_map.untyped_storage().data_ptr()!=t._expert_lookup.untyped_storage().data_ptr()
                inputs=self.contract_inputs(32,'i32_s32');values=routes(32,'hot_local')
                inputs[2].copy_(torch.tensor(values,dtype=torch.int32).reshape(32,8))
                initial=expert_mapping('contiguous');remote=expert_mapping('all_remote')
                captures={name:self.capture(lambda m=m:m(*inputs)) for name,m in [('two',t),('single',s)]} if self.gpu else {}
                def forward(name,m):
                    if name in captures:
                        graph,out,_=captures[name];graph.replay();return out
                    return m(*inputs)
                before=t._expert_map.data_ptr()
                self.check_result(forward('two',t),t,inputs,values,initial)
                t._expert_map.fill_(-1)
                stale=forward('two',t)
                assert any(v>=0 for v in stale[1].flatten().tolist()),'Expected stale cache was not observable'
                t.refresh_expert_lookup_()
                self.check_result(forward('two',t),t,inputs,values,remote)
                assert t._expert_map.data_ptr()==before
                t.update_expert_map_(torch.tensor(initial,dtype=torch.int32,device=inputs[2].device))
                self.check_result(forward('two',t),t,inputs,values,initial)
                s._expert_map.fill_(-1)
                self.check_result(forward('single',s),s,inputs,values,remote)
            return {'status':'passed','two_buffer_original_map_registered':True,'two_buffer_map_offset_elements':0,'single_owner_map_offset_elements':1,
                'two_buffer_direct_write_staleness_demonstrated':True,'explicit_refresh_restores_output':True,'checked_update_restores_output':True,
                'single_owner_direct_write_visible':True,'execution':'graph_replay' if self.gpu else 'eager_cpu',
                'note':'Mutation/refresh correctness only, excluded from static timings. Preserved registered-map storage is not proof of full framework or fused-kernel compatibility.'}
        return self.section('preservation_contract',run)

    def ready(self):return self.gpu and all(self.report['sections'].get(s,{}).get('status')=='passed' for s in ('lifecycle','input_contract','preservation_contract'))

    def make_cache_case(self,n,depth,pattern):
        inputs=self.contract_inputs(n,'i32_s32');values=routes(n,pattern)
        inputs[2].copy_(torch.tensor(values,dtype=torch.int32).reshape(n,8))
        chains={};variants={}
        def eager(chain):
            def fn():return tuple(m(*inputs) for m in chain)
            return fn
        def replay(capture):
            graph,out,_=capture
            def fn():graph.replay();return out
            return fn
        for name,cls in [('baseline',BaselineSourceSlice),('two_buffer',CandidateSourceSlice),('single_owner',SharedSourceSlice)]:
            with torch.inference_mode(False):chain=[cls().cuda() for _ in range(depth)]
            chains[name]=chain;fn=eager(chain)
            variants[name+'_eager']=fn;variants[name+'_graph']=replay(self.capture(fn))
        return chains,inputs,variants

    def check_cache_case(self,bundle,n,pattern):
        chains,inputs,variants=bundle;checks=0;mapping=expert_mapping('contiguous')
        for changed in ('sentinel','all_remote'):
            values=routes(n,changed);inputs[2].copy_(torch.tensor(values,dtype=torch.int32).reshape(n,8))
            for name,fn in variants.items():
                chain=chains[name.rsplit('_',1)[0]];out=fn();assert len(out)==len(chain)
                for output,m in zip(out,chain):
                    assert not self.check_result(output,m,inputs,values,mapping)
                    assert output[2].data_ptr()==inputs[1].data_ptr();checks+=1
        values=routes(n,pattern);inputs[2].copy_(torch.tensor(values,dtype=torch.int32).reshape(n,8))
        return checks,values,mapping

    def benchmark(self):
        def run():
            if not self.ready():return {'status':'skipped','reason':'Requires passing CUDA lifecycle, input and preservation gates'}
            with torch.inference_mode():
                for trial in range(self.trials):
                    configs=[(n,d,p) for n in (9,32,128,512) for d in (1,8) for p in ('hot_local','uniform')];random.Random(11200+trial).shuffle(configs)
                    for index,(n,depth,pattern) in enumerate(configs):
                        identity={'trial':trial,'case_order':index,'tokens':n,'pattern':pattern,'scope':'cache_choice','stages':depth};bundle=None
                        try:
                            bundle=self.make_cache_case(n,depth,pattern);checks,values,mapping=self.check_cache_case(bundle,n,pattern)
                            self.measure_updates(bundle[2],identity)
                            for name,fn in bundle[2].items():
                                for output,m in zip(fn(),bundle[0][name.rsplit('_',1)[0]]):self.check_result(output,m,bundle[1],values,mapping)
                            self.cases.append(dict(identity,status='passed',changed_input_output_checks=checks))
                        except Exception as exc:
                            self.rows=[r for r in self.rows if not all(r[k]==v for k,v in identity.items())]
                            self.raw=[r for r in self.raw if not all(r[k]==v for k,v in identity.items())]
                            self.cases.append(dict(identity,status='failed',error=repr(exc),traceback=traceback.format_exc(limit=6)))
                        finally:torch.cuda.synchronize();bundle=None
                        if (index+1)%4==0:self.save();print('Cache-choice trial',trial+1,index+1,'/',len(configs),flush=True)
            failed=sum(c['status']!='passed' for c in self.cases)
            return {'status':'passed' if not failed else 'partial','cases':len(self.cases),'failed_cases':failed,'changed_input_output_checks':sum(c.get('changed_input_output_checks',0) for c in self.cases)}
        return self.section('benchmark',run)

    def profile(self):
        def run():
            if not self.ready():return {'status':'skipped','reason':'Requires passing CUDA gates'}
            with torch.inference_mode():
                bundle=self.make_cache_case(128,8,'hot_local');self.check_cache_case(bundle,128,'hot_local');variants=bundle[2]
                for fn in variants.values():
                    for _ in range(5):fn()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as prof:
                    for i in range(3):
                        names=list(variants);random.Random(11300+i).shuffle(names)
                        for name in names:
                            with torch.profiler.record_function('mapping/'+name):variants[name]()
                        prof.step()
                    torch.cuda.synchronize()
                path=self.out/'cache_choice_trace.json';prof.export_chrome_trace(str(path));result=parse_trace(path,self.out)
                result['note']='Separate static trace: eight stages, 128 tokens, three cycles per ownership/schedule variant. No refresh in timings or trace. Missing graph correlation or CPU-only trace is partial.'
                return result
        return self.section('profiler',run)

    def export(self):
        comparisons=[]
        pairs=[(c+'_'+s,b+'_'+s) for s in ('eager','graph') for c,b in [('two_buffer','baseline'),('single_owner','baseline'),('single_owner','two_buffer')]]
        for case in self.cases:
            if case['status']!='passed':continue
            key={k:case[k] for k in ('trial','tokens','pattern','scope','stages')}
            rows={r['variant']:r for r in self.rows if all(r[k]==v for k,v in key.items())}
            for candidate,baseline in pairs:
                row=dict(key,candidate=candidate,baseline=baseline)
                for metric in ('event','wall','enqueue'):
                    b,c=rows[baseline][metric+'_us_median'],rows[candidate][metric+'_us_median']
                    row.update({metric+'_baseline_us':b,metric+'_candidate_us':c,metric+'_reduction_percent':100*(1-c/b),metric+'_saved_us':b-c,metric+'_saved_us_per_stage':(b-c)/case['stages']})
                comparisons.append(row)
        self.report['totals']={'cases':len(self.cases),'timing_rows':len(self.rows),'raw_samples':len(self.raw),'comparisons':len(comparisons)}
        self.save();write_csv(self.out/'comparisons.csv',comparisons)
        lines=['NOTEBOOK 11: CACHE OWNERSHIP CHOICE','Run: '+self.report['run_id'],'',self.report['scope'],'']
        for name in ('environment','lifecycle','input_contract','preservation_contract','benchmark','profiler'):lines.append(name+': '+self.report['sections'].get(name,{}).get('status','not_run'))
        lines+=['','Static int32 IDs / float32 scores. Six variants interleaved in each sample. Eager and one combined graph per chain, 1/8 stages, 9/32/128/512 tokens, hot-local and uniform routes.',
            'Two-buffer keeps _expert_map registered and at storage offset zero, but direct map edits leave lookup stale until explicit refresh. The separate gate demonstrates this and validates refresh before replay.',
            'Single-owner exposes _expert_map as a property view at offset one and reflects direct writes without refresh. Those API/storage changes still need framework and real-kernel validation.',
            'Maps stay fixed during timing. This experiment does not measure update cost or prove actual production update frequency. Neither draft is promoted.',
            'Report paired absolute savings and ratios; small observed differences do not establish equivalence. CUDA-event, wall and enqueue intervals overlap and cannot be added/subtracted for attribution.',
            'Independent prefixes are not real model layers. No full model, expert GEMM, actual alignment, small fused kernel, communication, startup or H200 validation.','']
        (self.out/'SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name!='checksums.json'}
        (self.out/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        archive=self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():z.write(p,p.name)
        print('\n'.join(lines));print('Download:',archive);return archive
