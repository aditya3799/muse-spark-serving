"""Diagnostic update API ablation; source slices and candidate draft unchanged."""
import hashlib,json,random,time,traceback,uuid,zipfile
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import torch
if 'OwnershipStudy' not in globals():
    from ownership_study import OwnershipStudy
    from source_slices import BaselineSourceSlice,SharedSourceSlice
    from timing_support import routes,expert_mapping,write_csv,parse_trace,stats

class BaselineCheckedSlice(BaselineSourceSlice):
    # Experimental matched API control: exact same decorated method object.
    update_expert_map_=SharedSourceSlice.update_expert_map_

class UpdateStudy(OwnershipStudy):
    def __init__(self,manifest,helper_source,output_root=None,force_cpu=False,samples=11,repeats=3,trials=3):
        self.manifest,self.helper_source=manifest,helper_source
        self.t,self.gpu,self.force_cpu=None,False,force_cpu
        self.samples,self.repeats,self.trials=samples,repeats,trials
        root=Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        self.out=root/('v4_update_overhead_'+run_id);self.out.mkdir(parents=True,exist_ok=False)
        self.rows,self.raw,self.cases=[],[],[]
        self.report={'schema_version':8,'run_id':run_id,'source_manifest':manifest,
            'scope':'Unchanged Notebook 07 source slices. Diagnostic host update API ablation, not a production patch or full-model execution.',
            'config':{'tokens':[32,128],'replays_per_cycle':[1,2,4,8],'trials':trials,'samples':samples,'repeats':repeats,'dtype':'int64'},
            'contract':'Direct-copy controls require prevalidated inputs and stable storage. Bind views after final placement; rebind and recapture after moves/replacement. No per-call safety validation in diagnostic direct paths.',
            'sections':{}}
        self.save()

    @staticmethod
    def validate_bound(module,view,incoming):
        current=module._expert_map
        if current is None or current.data_ptr()!=view.data_ptr() or current.device!=view.device:
            raise ValueError('Stale bound map view: rebind and recapture')
        if incoming.shape!=view.shape or incoming.dtype!=torch.int32 or incoming.device!=view.device:
            raise ValueError('Invalid incoming map contract')

    def update_contract(self):
        def run():
            device='cuda:0' if self.gpu else 'cpu'
            checks=[]
            with torch.inference_mode():
                assert BaselineCheckedSlice.update_expert_map_ is SharedSourceSlice.update_expert_map_
                for cls in (BaselineCheckedSlice,SharedSourceSlice):
                    module=cls().to(device);bound=module._expert_map
                    incoming=torch.tensor(expert_mapping('shared'),dtype=torch.int32,device=device)
                    self.validate_bound(module,bound,incoming)
                    for bad in (incoming[:10],incoming.float()):
                        before=bound.clone()
                        try:module.update_expert_map_(bad)
                        except ValueError:pass
                        else:raise AssertionError('Checked API accepted invalid input')
                        assert torch.equal(before,bound)
                    for kind in ('contiguous','shared','permuted'):
                        incoming.copy_(torch.tensor(expert_mapping(kind),dtype=torch.int32,device=device))
                        bound.fill_(-1);module.update_expert_map_(incoming)
                        assert torch.equal(bound,incoming)
                    checks.append({'class':cls.__name__,'checked_updates':3,'invalid_updates_rejected':2})
                # Demonstrate why an old view must not silently survive relocation.
                module=SharedSourceSlice().to(device);old=module._expert_map
                module._expert_lookup=module._expert_lookup.clone()
                before=module._expert_map.clone();old.fill_(-1)
                assert torch.equal(before,module._expert_map)
                try:self.validate_bound(module,old,incoming)
                except ValueError:pass
                else:raise AssertionError('Stale view not rejected')
                self.validate_bound(module,module._expert_map,incoming)
            return {'status':'passed','checked_method_identity':True,'controls':checks,'stale_view_rejected':True,
                'note':'Contract checks occur outside timing. A retained view is invalid after owner replacement or a device move; this is not an automatically safe production API.'}
        return self.section('update_contract',run)

    def ready(self):
        return self.gpu and all(self.report['sections'].get(s,{}).get('status')=='passed' for s in ('lifecycle','update_contract'))

    def make_update_case(self,n,k):
        with torch.inference_mode(False):
            modules={'baseline':BaselineCheckedSlice().cuda(),'single':SharedSourceSlice().cuda()}
        inputs=self.inputs(n);values=routes(n,'hot_local');mapping=expert_mapping('contiguous')
        inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(n,8))
        incoming=torch.tensor(mapping,dtype=torch.int32,device='cuda')
        views={name:m._expert_map for name,m in modules.items()}
        for name,m in modules.items():self.validate_bound(m,views[name],incoming);views[name].copy_(incoming)
        captures={name:self.capture(lambda m=m:m(*inputs)) for name,m in modules.items()}
        def static(name):
            graph,output,_=captures[name]
            def fn():
                for _ in range(k):graph.replay()
                return output
            return fn
        b,s=modules['baseline'],modules['single'];bound=views['single']
        replay_b,replay_s=static('baseline'),static('single')
        def baseline_direct():b._expert_map.copy_(incoming);return replay_b()
        def baseline_checked():b.update_expert_map_(incoming);return replay_b()
        def single_checked():s.update_expert_map_(incoming);return replay_s()
        def single_property():s._expert_map.copy_(incoming);return replay_s()
        def single_bound():bound.copy_(incoming);return replay_s()
        variants={'baseline_static':replay_b,'single_static':replay_s,'baseline_direct':baseline_direct,
            'baseline_checked':baseline_checked,'single_checked':single_checked,'single_property':single_property,'single_bound':single_bound}
        return modules,inputs,incoming,views,captures,variants

    def check_update_case(self,bundle,n):
        modules,inputs,incoming,views,captures,variants=bundle
        checked=0
        for kind in ('contiguous','shared','permuted'):
            # Each map has at least one local ID. Ensure poison is observably different.
            mapping=expert_mapping(kind);local=next(i for i,v in enumerate(mapping) if v>=0)
            values=routes(n,'sentinel');values[3]=local
            inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(n,8))
            incoming.copy_(torch.tensor(mapping,dtype=torch.int32))
            for name,fn in variants.items():
                owner='baseline' if name.startswith('baseline') else 'single';m=modules[owner];view=views[owner]
                self.validate_bound(m,view,incoming)
                view.fill_(-1)
                assert not torch.equal(view,incoming)
                if name.endswith('_static'):view.copy_(incoming)
                result=fn();self.check_result(result,m,inputs,values,mapping)
                assert torch.equal(view,incoming)
                checked+=1
        mapping=expert_mapping('contiguous');values=routes(n,'hot_local')
        inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(n,8))
        incoming.copy_(torch.tensor(mapping,dtype=torch.int32))
        for view in views.values():view.copy_(incoming)
        return checked,values,mapping

    def measure_updates(self,variants,identity):
        for fn in variants.values():
            for _ in range(10):fn()
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();b.record();b.synchronize()
        grouped=defaultdict(list);rng=random.Random(8120+identity['trial'])
        for sample in range(self.samples):
            order=list(variants);rng.shuffle(order)
            for position,name in enumerate(order):
                torch.cuda.synchronize();wall_begin=time.perf_counter_ns();a.record()
                enqueue_begin=time.perf_counter_ns()
                for _ in range(self.repeats):variants[name]()
                enqueue_end=time.perf_counter_ns();b.record();b.synchronize()
                wall_end=time.perf_counter_ns()
                row=dict(identity,variant=name,sample=sample,order_in_sample=position,repeats=self.repeats,
                    event_us=a.elapsed_time(b)*1000/self.repeats,
                    wall_us=(wall_end-wall_begin)/1000/self.repeats,
                    enqueue_us=(enqueue_end-enqueue_begin)/1000/self.repeats)
                self.raw.append(row);grouped[name].append(row)
        for name,rows in grouped.items():
            result=dict(identity,variant=name,samples=self.samples,repeats=self.repeats)
            for metric in ('event','wall','enqueue'):result.update({metric+'_us_'+key:v for key,v in stats([r[metric+'_us'] for r in rows]).items()})
            self.rows.append(result)

    def benchmark(self):
        def run():
            if not self.ready():return {'status':'skipped','reason':'Requires passing CUDA lifecycle and update-contract gates'}
            with torch.inference_mode():
                for trial in range(self.trials):
                    configs=[(n,k) for n in (32,128) for k in (1,2,4,8)];random.Random(8200+trial).shuffle(configs)
                    for index,(n,k) in enumerate(configs):
                        identity={'trial':trial,'case_order':index,'tokens':n,'dtype':'int64','pattern':'hot_local','scope':'update_api','replays_per_cycle':k};bundle=None
                        try:
                            bundle=self.make_update_case(n,k);checked,values,mapping=self.check_update_case(bundle,n)
                            self.measure_updates(bundle[-1],identity)
                            for name,fn in bundle[-1].items():self.check_result(fn(),bundle[0]['baseline' if name.startswith('baseline') else 'single'],bundle[1],values,mapping)
                            self.cases.append(dict(identity,status='passed',poisoned_state_checks=checked))
                        except Exception as exc:
                            self.rows=[r for r in self.rows if not all(r[key]==v for key,v in identity.items())]
                            self.raw=[r for r in self.raw if not all(r[key]==v for key,v in identity.items())]
                            self.cases.append(dict(identity,status='failed',error=repr(exc),traceback=traceback.format_exc(limit=6)))
                        finally:torch.cuda.synchronize();bundle=None
                    self.save();print('Update API trial',trial+1,'/',self.trials,flush=True)
            failed=sum(c['status']!='passed' for c in self.cases)
            return {'status':'passed' if not failed else 'partial','cases':len(self.cases),'failed_cases':failed,'poisoned_state_checks':sum(c.get('poisoned_state_checks',0) for c in self.cases)}
        return self.section('benchmark',run)

    def profile(self):
        def run():
            if not self.ready():return {'status':'skipped','reason':'Requires passing CUDA gates'}
            with torch.inference_mode():
                bundle=self.make_update_case(128,1);self.check_update_case(bundle,128)
                chosen={n:f for n,f in bundle[-1].items() if not n.endswith('_static')}
                for fn in chosen.values():
                    for _ in range(5):fn()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as prof:
                    for i in range(4):
                        names=list(chosen);random.Random(8300+i).shuffle(names)
                        for name in names:
                            with torch.profiler.record_function('mapping/'+name):chosen[name]()
                        prof.step()
                    torch.cuda.synchronize()
                path=self.out/'update_trace.json';prof.export_chrome_trace(str(path));result=parse_trace(path,self.out)
                result['note']='Separate profiler-on update-plus-graph schedule at K=1; only actual device operations counted. CPU-only capture or missing graph correlation is partial. Do not subtract trace durations from benchmark times.'
                return result
        return self.section('profiler',run)

    def export(self):
        comparisons=[]
        pairs=[('single_static','baseline_static'),('baseline_checked','baseline_direct'),('single_checked','baseline_direct'),
            ('single_property','baseline_direct'),('single_bound','baseline_direct'),('single_checked','baseline_checked'),
            ('single_property','single_checked'),('single_bound','single_property')]
        for case in self.cases:
            if case['status']!='passed':continue
            key={k:case[k] for k in ('trial','tokens','dtype','pattern','scope','replays_per_cycle')}
            rows={r['variant']:r for r in self.rows if all(r[k]==v for k,v in key.items())}
            for candidate,baseline in pairs:
                row=dict(key,candidate=candidate,baseline=baseline)
                for metric in ('event','wall','enqueue'):
                    b,c=rows[baseline][metric+'_us_median'],rows[candidate][metric+'_us_median']
                    row.update({metric+'_baseline_us':b,metric+'_candidate_us':c,metric+'_reduction_percent':100*(1-c/b)})
                comparisons.append(row)
        self.report['totals']={'cases':len(self.cases),'timing_rows':len(self.rows),'raw_samples':len(self.raw),'comparisons':len(comparisons)}
        self.save();write_csv(self.out/'comparisons.csv',comparisons)
        lines=['NOTEBOOK 08: UPDATE API OVERHEAD','Run: '+self.report['run_id'],'',self.report['scope'],'']
        for name in ('environment','lifecycle','update_contract','benchmark','profiler'):lines.append(name+': '+self.report['sections'].get(name,{}).get('status','not_run'))
        lines+=['','Timing rows are microseconds per complete update + K replays, or K replays for static controls.',
            'event: CUDA-event interval, which can include device idle gaps caused by host dispatch.',
            'wall: host elapsed time through the completion synchronization, including event-record overhead.',
            'enqueue: host time spent invoking update/replay calls, excluding event recording and final synchronization. It overlaps device execution; do not add it to or subtract it from event time.',
            'Baseline checked uses the identical decorated single-owner update method as a diagnostic control.',
            'Retained-view/direct-copy paths require prevalidated inputs and unchanged storage; they are not production proposals.',
            'Source slices are unchanged from Notebook 07. Actual alignment/GEMMs, offset-pointer fused kernels, full framework buffer handling and H200/full-serving performance remain untested.',
            'Inherited lifecycle includes all three Notebook 07 implementations; timed API ablation uses baseline and single-owner only.',
            'Negative reduction in comparisons.csv means slower candidate latency. Compare matched configurations and trials.','']
        (self.out/'SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name!='checksums.json'}
        (self.out/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        archive=self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():z.write(p,p.name)
        print('\n'.join(lines));print('Download:',archive);return archive
