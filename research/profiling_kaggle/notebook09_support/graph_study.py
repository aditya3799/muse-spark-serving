"""Static source-prefix stages under different graph boundaries; no model emulation."""
import hashlib,json,random,traceback,uuid,zipfile
from datetime import datetime,timezone
from pathlib import Path
import torch
if 'UpdateStudy' not in globals():
    from update_study import UpdateStudy
    from source_slices import BaselineSourceSlice,SharedSourceSlice
    from timing_support import routes,expert_mapping,write_csv,parse_trace

class GraphStudy(UpdateStudy):
    def __init__(self,manifest,helper_source,output_root=None,force_cpu=False,samples=11,repeats=3,trials=3):
        self.manifest,self.helper_source=manifest,helper_source
        self.t,self.gpu,self.force_cpu=None,False,force_cpu
        self.samples,self.repeats,self.trials=samples,repeats,trials
        root=Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        self.out=root/('v4_graph_boundaries_'+run_id);self.out.mkdir(parents=True,exist_ok=False)
        self.rows,self.raw,self.cases=[],[],[]
        self.report={'schema_version':9,'run_id':run_id,'source_manifest':manifest,
            'scope':'Static maps, independent source-prefix stages, three execution schedules. Not full model layers or a production graph.',
            'config':{'tokens':[9,32,128],'stages':[1,8,32],'trials':trials,'samples':samples,'repeats':repeats,'dtype':'int64','schedules':['eager','separate','combined']},
            'contract':'Maps and owning storage stay fixed after setup. Only routing ID values change between correctness checks. Each stage has a distinct module/map and all outputs are retained. No intervening compute or cross-stage data dependence.',
            'sections':{}}
        self.save()

    def ready(self):
        return self.gpu and self.report['sections'].get('lifecycle',{}).get('status')=='passed'

    def make_chain(self,n,depth):
        with torch.inference_mode(False):
            modules={name:[cls().cuda() for _ in range(depth)] for name,cls in [('baseline',BaselineSourceSlice),('single',SharedSourceSlice)]}
        inputs=self.inputs(n);values=routes(n,'hot_local')
        inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(n,8))
        mappings=[expert_mapping(('contiguous','shared','permuted')[i%3]) for i in range(depth)]
        for name,chain in modules.items():
            for m,mapping in zip(chain,mappings):
                incoming=torch.tensor(mapping,dtype=torch.int32,device='cuda')
                m._expert_map.copy_(incoming)
        def eager(chain):
            def fn():return tuple(m(*inputs) for m in chain)
            return fn
        def separate(captured):
            graphs=tuple(c[0] for c in captured);outputs=tuple(c[1] for c in captured)
            def fn():
                for g in graphs:g.replay()
                return outputs
            return fn
        def combined(captured):
            graph,output,_=captured
            def fn():graph.replay();return output
            return fn
        variants={};capture_ms={}
        for name,chain in modules.items():
            fn=eager(chain);each=[self.capture(lambda m=m:m(*inputs)) for m in chain];whole=self.capture(fn)
            variants[name+'_eager']=fn;variants[name+'_separate']=separate(each);variants[name+'_combined']=combined(whole)
            capture_ms[name+'_separate']=sum(c[2] for c in each);capture_ms[name+'_combined']=whole[2]
        return modules,inputs,mappings,variants,capture_ms

    def check_chain(self,bundle,n):
        modules,inputs,mappings,variants,_=bundle;checks=0
        addresses={name:[m._expert_map.data_ptr() for m in chain] for name,chain in modules.items()}
        for pattern in ('sentinel','hot_local'):
            values=routes(n,pattern);inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(n,8))
            for name,fn in variants.items():
                chain=modules[name.split('_')[0]];outputs=fn();assert len(outputs)==len(chain)
                for output,m,mapping in zip(outputs,chain,mappings):
                    assert not self.check_result(output,m,inputs,values,mapping);checks+=1
        for name,chain in modules.items():assert [m._expert_map.data_ptr() for m in chain]==addresses[name]
        return checks,values

    def benchmark(self):
        def run():
            if not self.ready():return {'status':'skipped','reason':'Requires passing CUDA lifecycle gate'}
            with torch.inference_mode():
                for trial in range(self.trials):
                    configs=[(n,d) for n in (9,32,128) for d in (1,8,32)];random.Random(9200+trial).shuffle(configs)
                    for index,(n,depth) in enumerate(configs):
                        identity={'trial':trial,'case_order':index,'tokens':n,'dtype':'int64','pattern':'hot_local','scope':'graph_boundary','stages':depth};bundle=None
                        try:
                            bundle=self.make_chain(n,depth);checked,values=self.check_chain(bundle,n)
                            self.measure_updates(bundle[3],identity)
                            for name,fn in bundle[3].items():
                                for output,m,mapping in zip(fn(),bundle[0][name.split('_')[0]],bundle[2]):self.check_result(output,m,bundle[1],values,mapping)
                            self.cases.append(dict(identity,status='passed',changed_input_output_checks=checked,capture_setup_ms=bundle[4]))
                        except Exception as exc:
                            self.rows=[r for r in self.rows if not all(r[key]==v for key,v in identity.items())]
                            self.raw=[r for r in self.raw if not all(r[key]==v for key,v in identity.items())]
                            self.cases.append(dict(identity,status='failed',error=repr(exc),traceback=traceback.format_exc(limit=6)))
                        finally:torch.cuda.synchronize();bundle=None
                        if (index+1)%3==0:self.save();print('Graph-boundary trial',trial+1,index+1,'/',len(configs),flush=True)
            failed=sum(c['status']!='passed' for c in self.cases)
            return {'status':'passed' if not failed else 'partial','cases':len(self.cases),'failed_cases':failed,'changed_input_output_checks':sum(c.get('changed_input_output_checks',0) for c in self.cases)}
        return self.section('benchmark',run)

    def profile(self):
        def run():
            if not self.ready():return {'status':'skipped','reason':'Requires passing CUDA gate'}
            with torch.inference_mode():
                bundle=self.make_chain(128,8);self.check_chain(bundle,128);variants=bundle[3]
                for fn in variants.values():
                    for _ in range(5):fn()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as prof:
                    for i in range(3):
                        names=list(variants);random.Random(9300+i).shuffle(names)
                        for name in names:
                            with torch.profiler.record_function('mapping/'+name):variants[name]()
                        prof.step()
                    torch.cuda.synchronize()
                path=self.out/'graph_boundary_trace.json';prof.export_chrome_trace(str(path));result=parse_trace(path,self.out)
                result['note']='Separate trace: eight static stages, 128 tokens, three cycles per variant. Count actual kernels/copies only; missing correlation yields partial status. No map updates. Compare operation counts across graph boundaries; never subtract trace durations from benchmark time.'
                return result
        return self.section('profiler',run)

    def export(self):
        comparisons=[]
        pairs=[('single_'+s,'baseline_'+s) for s in ('eager','separate','combined')]
        pairs += [(v+'_combined',v+'_separate') for v in ('baseline','single')]
        for case in self.cases:
            if case['status']!='passed':continue
            key={k:case[k] for k in ('trial','tokens','dtype','pattern','scope','stages')}
            rows={r['variant']:r for r in self.rows if all(r[k]==v for k,v in key.items())}
            for candidate,baseline in pairs:
                row=dict(key,candidate=candidate,baseline=baseline)
                for metric in ('event','wall','enqueue'):
                    b,c=rows[baseline][metric+'_us_median'],rows[candidate][metric+'_us_median']
                    row.update({metric+'_baseline_us':b,metric+'_candidate_us':c,metric+'_reduction_percent':100*(1-c/b),metric+'_baseline_us_per_stage':b/case['stages'],metric+'_candidate_us_per_stage':c/case['stages']})
                comparisons.append(row)
        self.report['totals']={'cases':len(self.cases),'timing_rows':len(self.rows),'raw_samples':len(self.raw),'comparisons':len(comparisons)}
        self.save();write_csv(self.out/'comparisons.csv',comparisons)
        lines=['NOTEBOOK 09: STATIC MAPPING AND GRAPH BOUNDARIES','Run: '+self.report['run_id'],'',self.report['scope'],'']
        for name in ('environment','lifecycle','benchmark','profiler'):lines.append(name+': '+self.report['sections'].get(name,{}).get('status','not_run'))
        lines+=['','Each sample measures complete chains of 1/8/32 independent pre-alignment source prefixes. Per-stage numbers divide total time; they are not separately measured stage latency.',
            'Same fixed maps and input shapes across eager, N separate graphs, and one combined graph. Captured graphs allocate separate outputs, and all outputs are retained. No map updates or device moves occur during timing.',
            'Host enqueue, CUDA-event and synchronized wall times overlap; do not add/subtract them for CPU/GPU attribution.',
            'Stages are not actual transformer layers and are not data-dependent. There is no attention, expert GEMM, collectives, alignment or filler compute. This does not model full-serving speedup.',
            'Source/drafts unchanged. Small fused branch remains untested; token counts start at nine for top-k eight. Initialization/tokenizer profiling is outside this experiment.','']
        (self.out/'SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name!='checksums.json'}
        (self.out/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        archive=self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():z.write(p,p.name)
        print('\n'.join(lines));print('Download:',archive);return archive
