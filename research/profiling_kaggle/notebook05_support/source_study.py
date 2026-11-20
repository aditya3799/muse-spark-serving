"""Execute source-selected mapping prefixes and measure update/reuse sequences."""
import hashlib
import json
import random
import traceback
import uuid
import zipfile
from datetime import datetime,timezone
from pathlib import Path
import torch

if 'ExperimentCore' not in globals():
    from timing_support import ExperimentCore,routes,reference,expert_mapping,write_csv,parse_trace
if 'BaselineSourceSlice' not in globals():
    from source_slices import BaselineSourceSlice,CandidateSourceSlice


class SourceStudy(ExperimentCore):
    def __init__(self,manifest,helper_source,output_root=None,force_cpu=False,samples=11,repeats=20,trials=3):
        self.manifest,self.helper_source=manifest,helper_source
        self.t,self.gpu,self.force_cpu=None,False,force_cpu
        self.samples,self.repeats,self.trials=samples,repeats,trials
        root=Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        self.out=root/('v4_source_candidate_'+run_id);self.out.mkdir(parents=True,exist_ok=False)
        self.rows,self.raw,self.cases=[],[],[]
        self.report={'schema_version':5,'run_id':run_id,'source_manifest':manifest,
            'scope':'AST-selected real-source initialization and pre-alignment prefix, comparing V4 with an unapplied draft. No full serving class, aligner, GEMM or H200 execution.',
            'config':{'samples':samples,'prefix_repeats':repeats,'trials':trials,'prefix_tokens':[9,32,128,256],'cadence_tokens':[32,128],'calls_per_update':[1,8,64],'cadence_repeats':3},
            'timing_units':'source_prefix: us per prefix call. update_cadence: us per update-plus-K-prefix cycle; comparisons also normalize cycle duration by K. No separately measured costs are added.',
            'sections':{}}
        self.save()

    def pair(self,no_map=False):
        kwargs={'num_experts':256,'num_local_experts':256 if no_map else 32,'local_expert_offset':0 if no_map else 64}
        device='cuda:0' if self.gpu else 'cpu'
        return BaselineSourceSlice(**kwargs).to(device),CandidateSourceSlice(**kwargs).to(device)

    def inputs(self,n,dtype='int64',strided=False):
        device='cuda:0' if self.gpu else 'cpu'
        raw=torch.empty((n,16 if strided else 8),dtype=getattr(torch,dtype),device=device)
        raw=raw[:,::2] if strided else raw
        return (torch.ones((n,32),device=device,dtype=torch.float16),torch.full((n,8),0.125,device=device,dtype=torch.float16),raw)

    def set_inputs(self,pair,inputs,n,pattern='uniform',kind='contiguous',seed=604):
        values=routes(n,pattern,seed);mapping=expert_mapping(kind,seed)
        inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(n,8))
        if pair[0]._expert_map is not None:
            new=torch.tensor(mapping,dtype=torch.int32,device=inputs[2].device)
            pair[0]._expert_map.copy_(new);pair[1].update_expert_map_(new)
        return values,mapping if pair[0]._expert_map is not None else None

    def check_result(self,output,module,inputs,values,mapping):
        fused,local,weights=output
        ids=inputs[2]
        expected_fused=(mapping is not None and inputs[0].is_cuda and ids.is_contiguous() and module._expert_map.is_contiguous() and 0<ids.numel()<=64)
        assert fused==expected_fused,'Source branch changed'
        assert local.dtype==torch.int32 and local.shape==ids.shape and local.device==ids.device
        assert weights.dtype==torch.float32 and weights.shape==ids.shape and weights.is_contiguous()
        assert weights.flatten().tolist()==[0.125]*len(values)
        if not fused:
            assert local.is_contiguous()
            assert local.flatten().tolist()==(values if mapping is None else reference(values,mapping))
        # In fused branch the source allocates local IDs for a later kernel.
        # Never inspect uninitialized values or claim that kernel was tested.
        return fused

    def source_checks(self):
        def run():
            counts={'input_cases':0,'fallback_cases':0,'deferred_fused_cases':0}
            for kind in ('contiguous','permuted','shared','all_remote'):
                pair=self.pair()
                for n in (0,1,8,9,32,128,256):
                    for dtype in ('int32','int64'):
                        for strided in (False,True):
                            inputs=self.inputs(n,dtype,strided)
                            for pattern in ('uniform','hot_local','all_local','all_remote','sentinel','all_negative'):
                                values,mapping=self.set_inputs(pair,inputs,n,pattern,kind)
                                branches=[self.check_result(m(*inputs),m,inputs,values,mapping) for m in pair]
                                assert branches[0]==branches[1]
                                counts['input_cases']+=1
                                counts['deferred_fused_cases' if branches[0] else 'fallback_cases']+=1
            pair=self.pair(no_map=True)
            for n in (0,1,8,9,128):
                inputs=self.inputs(n);values,mapping=self.set_inputs(pair,inputs,n,'sentinel')
                for m in pair:self.check_result(m(*inputs),m,inputs,values,mapping)
            # Noncontiguous expert-map storage must also preserve the original guard.
            pair=self.pair();inputs=self.inputs(8)
            values,mapping=self.set_inputs(pair,inputs,8,'sentinel')
            for m in pair:
                storage=torch.empty(512,dtype=torch.int32,device=m._expert_map.device)
                storage[::2].copy_(m._expert_map);m._expert_map=storage[::2]
            pair[1].refresh_expert_lookup_()
            for m in pair:assert not self.check_result(m(*inputs),m,inputs,values,mapping)
            return {'status':'passed',**counts,'no_map_cases':5,'noncontiguous_map_guard_cases':1,
                'note':'Deferred fused cases validate only branch, buffer shape/dtype and scores. Real small-route alignment is not executed. Fallback results are checked against independent CPU mapping.'}
        return self.section('source_checks',run)

    def benchmark(self):
        def run():
            if not self.gpu or self.report['sections'].get('source_checks',{}).get('status')!='passed':
                return {'status':'skipped','reason':'Requires passing CUDA source checks'}
            configs=[(n,d,p) for n in (9,32,128,256) for d in ('int32','int64') for p in ('uniform','hot_local')]
            for trial in range(self.trials):
                order=list(configs);random.Random(2100+trial).shuffle(order)
                for index,(n,dtype,pattern) in enumerate(order):
                    identity={'trial':trial,'case_order':index,'tokens':n,'dtype':dtype,'pattern':pattern,'scope':'source_prefix','calls_per_update':0}
                    captures={};variants={}
                    try:
                        pair=self.pair();inputs=self.inputs(n,dtype)
                        values,mapping=self.set_inputs(pair,inputs,n,pattern)
                        fns={name:(lambda m=m:m(*inputs)) for name,m in zip(('baseline','candidate'),pair)}
                        captures={name:self.capture(fn) for name,fn in fns.items()}
                        for changed_pattern,kind in [('sentinel','permuted'),('all_negative','shared'),('all_local','all_remote')]:
                            values,mapping=self.set_inputs(pair,inputs,n,changed_pattern,kind)
                            for name,m in zip(fns,pair):
                                graph,output,_=captures[name];graph.replay()
                                self.check_result(output,m,inputs,values,mapping)
                        values,mapping=self.set_inputs(pair,inputs,n,pattern)
                        for name,m in zip(fns,pair):
                            graph,output,_=captures[name];graph.replay();self.check_result(output,m,inputs,values,mapping)
                            variants[name+'_eager']=fns[name];variants[name+'_graph']=graph.replay
                        self.measure(variants,identity)
                        for name,m in zip(fns,pair):
                            self.check_result(fns[name](),m,inputs,values,mapping)
                            graph,output,_=captures[name];graph.replay();self.check_result(output,m,inputs,values,mapping)
                        self.cases.append(dict(identity,status='passed',changed_input_graph_checks=6))
                    except Exception as exc:
                        self.drop_case(identity,exc)
                    finally:
                        torch.cuda.synchronize();variants.clear();captures.clear()
                    if (index+1)%8==0:self.save();print('Prefix trial',trial+1,index+1,'/',len(order),flush=True)
            failed=sum(c['status']!='passed' for c in self.cases)
            return {'status':'passed' if not failed else 'partial','cases':len(self.cases),'failed_cases':failed,'changed_input_graph_checks':sum(c.get('changed_input_graph_checks',0) for c in self.cases)}
        return self.section('benchmark',run)

    def drop_case(self,identity,exc):
        self.rows=[r for r in self.rows if not all(r[k]==v for k,v in identity.items())]
        self.raw=[r for r in self.raw if not all(r[k]==v for k,v in identity.items())]
        self.cases.append(dict(identity,status='failed',error=repr(exc),traceback=traceback.format_exc(limit=6)))

    def cadence(self):
        def run():
            if not self.gpu or self.report['sections'].get('source_checks',{}).get('status')!='passed':
                return {'status':'skipped','reason':'Requires validated CUDA source slices'}
            old_repeats=self.repeats;self.repeats=3
            first=len(self.cases)
            try:
                for trial in range(self.trials):
                    configs=[(n,k) for n in (32,128) for k in (1,8,64)];random.Random(2300+trial).shuffle(configs)
                    for index,(n,k) in enumerate(configs):
                        identity={'trial':trial,'case_order':index,'tokens':n,'dtype':'int64','pattern':'hot_local','scope':'update_cadence','calls_per_update':k}
                        captures={};variants={}
                        try:
                            pair=self.pair();inputs=self.inputs(n)
                            values,mapping=self.set_inputs(pair,inputs,n,'hot_local')
                            incoming=torch.tensor(mapping,dtype=torch.int32,device='cuda')
                            def make_cycle(module,is_candidate):
                                def cycle():
                                    if is_candidate:module.update_expert_map_(incoming)
                                    else:module._expert_map.copy_(incoming)
                                    for _ in range(k):result=module(*inputs)
                                    return result
                                return cycle
                            fns={'baseline':make_cycle(pair[0],False),'candidate':make_cycle(pair[1],True)}
                            captures={name:self.capture(fn) for name,fn in fns.items()}
                            for kind in ('permuted','shared'):
                                mapping=expert_mapping(kind)
                                incoming.copy_(torch.tensor(mapping,dtype=torch.int32))
                                for name,m in zip(fns,pair):
                                    graph,output,_=captures[name];graph.replay();self.check_result(output,m,inputs,values,mapping)
                            mapping=expert_mapping();incoming.copy_(torch.tensor(mapping,dtype=torch.int32))
                            for name,m in zip(fns,pair):
                                graph,output,_=captures[name];graph.replay();self.check_result(output,m,inputs,values,mapping)
                                variants[name+'_eager']=fns[name];variants[name+'_graph']=graph.replay
                            self.measure(variants,identity)
                            for name,m in zip(fns,pair):
                                self.check_result(fns[name](),m,inputs,values,mapping)
                                graph,output,_=captures[name];graph.replay();self.check_result(output,m,inputs,values,mapping)
                            self.cases.append(dict(identity,status='passed',changed_input_graph_checks=4))
                        except Exception as exc:self.drop_case(identity,exc)
                        finally:torch.cuda.synchronize();variants.clear();captures.clear()
                    self.save();print('Update cadence trial',trial+1,'complete',flush=True)
            finally:self.repeats=old_repeats
            selected=self.cases[first:];failed=sum(c['status']!='passed' for c in selected)
            return {'status':'passed' if not failed else 'partial','cases':len(selected),'failed_cases':failed,'changed_mapping_graph_checks':sum(c.get('changed_input_graph_checks',0) for c in selected),
                    'note':'A cycle copies updated map values, then runs K real-source prefixes. Candidate refresh is inside the timed cycle. Graph captures the entire cycle, not K separate replay calls. No serving workload or actual update-frequency assumption.'}
        return self.section('update_cadence',run)

    def profile(self):
        def run():
            if not self.gpu or self.report['sections'].get('source_checks',{}).get('status')!='passed':return {'status':'skipped','reason':'Requires validated CUDA source slices'}
            pair=self.pair();inputs=self.inputs(128);values,mapping=self.set_inputs(pair,inputs,128,'hot_local')
            fns={name:(lambda m=m:m(*inputs)) for name,m in zip(('baseline_source','candidate_source'),pair)}
            for fn in fns.values():
                for _ in range(10):fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as prof:
                for i in range(6):
                    names=list(fns);random.Random(170+i).shuffle(names)
                    for name in names:
                        with torch.profiler.record_function('mapping/'+name):fns[name]()
                    prof.step()
                torch.cuda.synchronize()
            path=self.out/'mapping_trace.json';prof.export_chrome_trace(str(path))
            return parse_trace(path,self.out)
        return self.section('profiler',run)

    def export(self):
        comparisons=[]
        for case in self.cases:
            if case['status']!='passed':continue
            key={k:case[k] for k in ('trial','tokens','dtype','pattern','scope','calls_per_update')}
            rows={r['variant']:r for r in self.rows if all(r[k]==v for k,v in key.items())}
            for mode in ('eager','graph'):
                b,c=rows['baseline_'+mode],rows['candidate_'+mode];k=case['calls_per_update'] or 1
                comparisons.append(dict(key,execution=mode,baseline_event_us=b['event_us_median'],candidate_event_us=c['event_us_median'],
                    baseline_over_candidate=b['event_us_median']/c['event_us_median'],baseline_us_per_mapping_including_update=b['event_us_median']/k,candidate_us_per_mapping_including_update=c['event_us_median']/k,
                    wall_saved_us=b['wall_us_median']-c['wall_us_median']))
        self.report['totals']={'cases':len(self.cases),'timing_rows':len(self.rows),'raw_samples':len(self.raw)}
        self.save();write_csv(self.out/'comparisons.csv',comparisons)
        lines=['NOTEBOOK 05: SOURCE DRAFT AND UPDATE CADENCE','Run: '+self.report['run_id'],'',self.report['scope'],'']
        for name in ('environment','source_checks','benchmark','update_cadence','profiler'):lines.append(name+': '+self.report['sections'].get(name,{}).get('status','not_run'))
        lines+=['','UPDATE-PLUS-K-PREFIX CYCLES (normalized us per mapping, including update share)','Trial Tokens K Mode Baseline Candidate Ratio']
        for r in comparisons:
            if r['scope']=='update_cadence':lines.append('{trial} {tokens} {calls_per_update} {execution} {baseline_us_per_mapping_including_update:.2f} {candidate_us_per_mapping_including_update:.2f} {baseline_over_candidate:.3f}'.format(**r))
        lines+=['','INTERPRETATION',
            'Baseline and draft initialization/prefix statements were selected directly from source by AST. Nonportable imports and the entire aligner/GEMM tail are omitted, not stubbed into performance results.',
            'For <=64-route fused cases, local IDs are uninitialized until the omitted kernel. Only branch/metadata and top-k scores are checked. These cases are not timed.',
            'Source-prefix timings include hidden-state contiguity, source dispatch guard, conversion/mapping and score conversion, before real alignment. They are not complete MoE latency.',
            'Cadence timings execute one update followed by K prefix calls in the SAME timed operation. Candidate refresh is included. Graph execution captures the complete cycle; this is a synthetic replay workload, not traffic inference.',
            'Both correctness and timing use fixed buffer shapes. Direct map writes still require explicit refresh; relocation requires recapture. Runtime update frequency is unmeasured.',
            'The V4-relative draft is unapplied. Static checks verify the fused guard and alignment/GEMM tail are unchanged; this does not validate real model execution.',
            'Read all three trials and raw samples. CUDA-event intervals include gaps; profiling is separate. No H200 or full-serving speedup claim.','']
        (self.out/'SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')
        checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name!='checksums.json'}
        (self.out/'checksums.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        archive=self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():z.write(p,p.name)
        print('\n'.join(lines));print('Download:',archive);return archive
