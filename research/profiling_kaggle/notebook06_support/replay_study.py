"""Inference-mode replay scheduling; unchanged V4-relative source candidate."""
import hashlib,json,random,traceback,uuid,zipfile
from datetime import datetime,timezone
from pathlib import Path
import torch
if 'SourceInputs' not in globals():
    from input_harness import SourceInputs
    from timing_support import routes,reference,expert_mapping,write_csv,parse_trace


class ReplayStudy(SourceInputs):
    def __init__(self,manifest,helper_source,output_root=None,force_cpu=False,samples=11,repeats=3,trials=3):
        self.manifest,self.helper_source=manifest,helper_source
        self.t,self.gpu,self.force_cpu=None,False,force_cpu
        self.samples,self.repeats,self.trials=samples,repeats,trials
        root=Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        self.out=root/('v4_replay_schedule_'+run_id);self.out.mkdir(parents=True,exist_ok=False)
        self.rows,self.raw,self.cases=[],[],[]
        self.report={'schema_version':6,'run_id':run_id,'source_manifest':manifest,
            'scope':'Same unapplied source-slice candidate as Notebook 05, under inference mode. Compare host-side updates outside existing prefix graphs with static maps and fully captured update cycles. No actual model/aligner/GEMM execution.',
            'config':{'tokens':[32,128],'replays_per_cycle':[1,2,4,8,64],'trials':trials,'samples':samples,'repeats':repeats,'dtype':'int64','pattern':'hot_local'},
            'scenarios':{'static':'No map update; K separate prefix-graph launches.','external':'Eager update API outside capture, then K separate prefix-graph launches.','captured':'One replay of a graph containing the update and K prefixes; control from Notebook 05.'},
            'units':'Every timing row is us per complete cycle. Comparison rows also divide by K to include the update share per prefix. Compare candidate versus baseline within each scenario.',
            'sections':{}}
        self.save()

    def correctness(self):
        def run():
            if self.t is None:return {'status':'skipped','reason':'PyTorch unavailable'}
            count=0
            with torch.inference_mode():
                for created_in_inference in (False,True):
                    with torch.inference_mode(created_in_inference):pair=self.pair()
                    assert all(m._expert_map.is_inference()==created_in_inference for m in pair)
                    for n in (9,32,128,256):
                        for dtype in ('int32','int64'):
                            inputs=self.inputs(n,dtype)
                            for kind in ('contiguous','permuted','shared'):
                                for pattern in ('hot_local','sentinel','all_negative','all_remote'):
                                    values,mapping=self.set_inputs(pair,inputs,n,pattern,kind)
                                    for m in pair:
                                        output=m(*inputs)
                                        assert not self.check_result(output,m,inputs,values,mapping)
                                    count+=1
            return {'status':'passed','input_cases':count,'output_checks':count*2,'creation_modes':['normal','inference'],
                'note':'All updates/forwards run under inference_mode. This does not establish that inference-created tensors can be mutated outside that context. Same draft and source slices as Notebook 05.'}
        return self.section('correctness',run)

    def make_case(self,n,k):
        assert torch.is_inference_mode_enabled()
        # Typical ownership: ordinary buffers created before inference begins.
        with torch.inference_mode(False):pair=self.pair()
        inputs=self.inputs(n)
        values,mapping=self.set_inputs(pair,inputs,n,'hot_local')
        incoming=torch.tensor(mapping,dtype=torch.int32,device='cuda')
        forwards={name:self.capture(lambda m=m:m(*inputs)) for name,m in zip(('baseline','candidate'),pair)}
        variants,cycles={},{}
        def update(module,is_candidate):
            if is_candidate:module.update_expert_map_(incoming)
            else:module._expert_map.copy_(incoming)
        def make_static(graph,output):
            def fn():
                for _ in range(k):graph.replay()
                return output
            return fn
        def make_external(module,is_candidate,static):
            def fn():
                update(module,is_candidate)
                return static()
            return fn
        def make_captured_body(module,is_candidate):
            def fn():
                update(module,is_candidate)
                for _ in range(k):output=module(*inputs)
                return output
            return fn
        def make_replay(graph,output):
            def fn():
                graph.replay()
                return output
            return fn
        for name,m in zip(forwards,pair):
            is_candidate=name=='candidate'
            graph,output,_=forwards[name]
            static=make_static(graph,output)
            variants[name+'_static']=static
            variants[name+'_external']=make_external(m,is_candidate,static)
            cycles[name]=self.capture(make_captured_body(m,is_candidate))
            variants[name+'_captured']=make_replay(cycles[name][0],cycles[name][1])
        return pair,inputs,incoming,forwards,cycles,variants

    def check_case(self,bundle,n):
        pair,inputs,incoming,forwards,cycles,variants=bundle
        checks=0
        for kind,pattern in [('permuted','sentinel'),('shared','all_negative')]:
            values,mapping=self.set_inputs(pair,inputs,n,pattern,kind)
            incoming.copy_(torch.tensor(mapping,dtype=torch.int32))
            for name,fn in variants.items():
                m=pair[1 if name.startswith('candidate') else 0]
                self.check_result(fn(),m,inputs,values,mapping);checks+=1
        self.restore_case(bundle,n)
        return checks

    def restore_case(self,bundle,n):
        pair,inputs,incoming,_,_,_=bundle
        values,mapping=self.set_inputs(pair,inputs,n,'hot_local')
        incoming.copy_(torch.tensor(mapping,dtype=torch.int32))
        return values,mapping

    def benchmark(self):
        def run():
            if not self.gpu or self.report['sections'].get('correctness',{}).get('status')!='passed':return {'status':'skipped','reason':'Requires passing CUDA correctness'}
            with torch.inference_mode():
                for trial in range(self.trials):
                    configs=[(n,k) for n in (32,128) for k in (1,2,4,8,64)]
                    random.Random(2800+trial).shuffle(configs)
                    for index,(n,k) in enumerate(configs):
                        identity={'trial':trial,'case_order':index,'tokens':n,'dtype':'int64','pattern':'hot_local','scope':'replay_schedule','replays_per_cycle':k}
                        bundle=None
                        try:
                            bundle=self.make_case(n,k)
                            checked=self.check_case(bundle,n)
                            pair,inputs,_,_,_,variants=bundle
                            values,mapping=self.restore_case(bundle,n)
                            # All variants use identical mapping values during timing.
                            # Static remains correct regardless of preceding variant order.
                            self.measure(variants,identity)
                            for name,fn in variants.items():
                                self.check_result(fn(),pair[1 if name.startswith('candidate') else 0],inputs,values,mapping)
                            self.cases.append(dict(identity,status='passed',changed_input_schedule_checks=checked))
                        except Exception as exc:
                            self.rows=[r for r in self.rows if not all(r[key]==value for key,value in identity.items())]
                            self.raw=[r for r in self.raw if not all(r[key]==value for key,value in identity.items())]
                            self.cases.append(dict(identity,status='failed',error=repr(exc),traceback=traceback.format_exc(limit=6)))
                        finally:
                            torch.cuda.synchronize();bundle=None
                        if (index+1)%5==0:self.save();print('Replay trial',trial+1,index+1,'/',len(configs),flush=True)
            failed=sum(c['status']!='passed' for c in self.cases)
            return {'status':'passed' if not failed else 'partial','cases':len(self.cases),'failed_cases':failed,
                'changed_input_schedule_checks':sum(c.get('changed_input_schedule_checks',0) for c in self.cases),
                'note':'Static/external launch K prefix graphs separately; captured launches one larger graph. Candidate/baseline ratios are comparable within each scenario. Cross-scenario differences also include launch topology.'}
        return self.section('benchmark',run)

    def profile(self):
        def run():
            if not self.gpu or self.report['sections'].get('correctness',{}).get('status')!='passed':return {'status':'skipped','reason':'Requires validated CUDA source slices'}
            with torch.inference_mode():
                bundle=self.make_case(128,4);self.check_case(bundle,128)
                pair,inputs,_,_,_,variants=bundle
                chosen={name:variants[name] for name in ('baseline_external','candidate_external')}
                for fn in chosen.values():
                    for _ in range(5):fn()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as prof:
                    for i in range(4):
                        names=list(chosen);random.Random(810+i).shuffle(names)
                        for name in names:
                            with torch.profiler.record_function('mapping/'+name):chosen[name]()
                        prof.step()
                    torch.cuda.synchronize()
                path=self.out/'replay_trace.json';prof.export_chrome_trace(str(path))
                result=parse_trace(path,self.out)
                result['graph_trace_note']='Graph kernels may lack host launch correlation on some profiler versions. Unattributed events remain in the CSV and make status partial; do not invent stage totals. Timing samples are independent.'
                return result
        return self.section('profiler',run)

    def export(self):
        comparisons=[]
        for case in self.cases:
            if case['status']!='passed':continue
            keys={k:case[k] for k in ('trial','tokens','dtype','pattern','scope','replays_per_cycle')}
            rows={r['variant']:r for r in self.rows if all(r[k]==v for k,v in keys.items())}
            for scenario in ('static','external','captured'):
                b,c=rows['baseline_'+scenario],rows['candidate_'+scenario];k=case['replays_per_cycle']
                comparisons.append(dict(keys,scenario=scenario,baseline_cycle_us=b['event_us_median'],candidate_cycle_us=c['event_us_median'],
                    baseline_over_candidate=b['event_us_median']/c['event_us_median'],baseline_us_per_replay=b['event_us_median']/k,candidate_us_per_replay=c['event_us_median']/k,
                    wall_saved_us=b['wall_us_median']-c['wall_us_median']))
        self.report['totals']={'cases':len(self.cases),'timing_rows':len(self.rows),'raw_samples':len(self.raw)}
        self.save();write_csv(self.out/'comparisons.csv',comparisons)
        lines=['NOTEBOOK 06: INFERENCE-MODE REPLAY SCHEDULE','Run: '+self.report['run_id'],'',self.report['scope'],'']
        for name in ('environment','correctness','benchmark','profiler'):lines.append(name+': '+self.report['sections'].get(name,{}).get('status','not_run'))
        lines+=['','EXTERNAL UPDATE + K SEPARATE PREFIX GRAPH REPLAYS','Trial Tokens K Baseline us/replay Candidate us/replay Ratio']
        for r in comparisons:
            if r['scenario']=='external':lines.append('{trial} {tokens} {replays_per_cycle} {baseline_us_per_replay:.2f} {candidate_us_per_replay:.2f} {baseline_over_candidate:.3f}'.format(**r))
        lines+=['','INTERPRETATION',
            'Primary comparison: external scenario, baseline versus candidate. Refresh happens as an eager operation outside the existing prefix graph, then K separate graph launches run.',
            'Static scenario has the same K separate launches but no update. Captured scenario records the update plus K prefixes into one graph; it is a control, not the same launch schedule.',
            'All calls run under torch.inference_mode. Benchmark module buffers were created normally; correctness also covers inference-created buffers updated within inference mode.',
            'Input generation/copies, graph capture and correctness checks are excluded. External scenario includes map copy and candidate refresh; incoming map data already resides on GPU.',
            'Each cycle performs the stated update even when values repeat. Production mapping frequency is not measured or inferred.',
            'Event intervals can include launch gaps; wall samples include host dispatch and synchronization. Cycles contain only source-sliced prefixes, not a real full-model graph.',
            'Compare within scenarios and trials. No H200/serving speedup or universal reuse threshold follows from these timings.',
            'Unchanged draft remains unapplied. No real alignment/GEMMs, scheduler/tokenizer startup or full model loading are tested.',
            'Direct map writes still need explicit refresh, inference-created tensors need compatible update context, and buffer relocation requires graph recapture.','']
        (self.out/'SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name!='checksums.json'}
        (self.out/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        archive=self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():z.write(p,p.name)
        print('\n'.join(lines));print('Download:',archive);return archive
