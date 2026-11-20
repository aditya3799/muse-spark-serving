"""Single-owner lookup: alias correctness and external-update comparison."""
import hashlib,io,json,random,traceback,uuid,zipfile
from datetime import datetime,timezone
from pathlib import Path
import torch
if 'SourceInputs' not in globals():
    from input_harness import SourceInputs
    from timing_support import routes,reference,expert_mapping,write_csv,parse_trace
    from source_slices import BaselineSourceSlice,CandidateSourceSlice,SharedSourceSlice


class OwnershipStudy(SourceInputs):
    def __init__(self,manifest,helper_source,output_root=None,force_cpu=False,samples=11,repeats=3,trials=3):
        self.manifest,self.helper_source=manifest,helper_source
        self.t,self.gpu,self.force_cpu=None,False,force_cpu
        self.samples,self.repeats,self.trials=samples,repeats,trials
        root=Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        self.out=root/('v4_single_owner_'+run_id);self.out.mkdir(parents=True,exist_ok=False)
        self.rows,self.raw,self.cases=[],[],[]
        self.report={'schema_version':7,'run_id':run_id,'source_manifest':manifest,
            'scope':'Source-sliced V4 baseline, prior two-buffer cache, and unapplied single-owner cache. Same lookup arithmetic; ownership/update mechanism differs. No real aligner/GEMM/full-model/H200 execution.',
            'config':{'tokens':[32,128],'replays_per_cycle':[1,2,4,8,64],'trials':trials,'samples':samples,'repeats':repeats,'dtype':'int64','scenarios':['static','external']},
            'contract':'Single-owner _expert_map is a property returning lookup[1:], not a registered buffer. In-place writes update both views; attribute replacement is rejected. Device moves recreate the view on access. Graphs still require recapture after relocation. Map storage offset is one int32 element.',
            'sections':{}}
        self.save()

    def modules(self,no_map=False,device=None):
        kwargs={'num_experts':256,'num_local_experts':256 if no_map else 32,'local_expert_offset':0 if no_map else 64}
        device=device or ('cuda:0' if self.gpu else 'cpu')
        return {'baseline':BaselineSourceSlice(**kwargs).to(device),'two_buffer':CandidateSourceSlice(**kwargs).to(device),'single_owner':SharedSourceSlice(**kwargs).to(device)}

    def update(self,name,module,incoming):
        if name=='baseline':module._expert_map.copy_(incoming)
        else:module.update_expert_map_(incoming)

    def load(self,modules,inputs,n,pattern='hot_local',kind='contiguous'):
        values,mapping=routes(n,pattern),expert_mapping(kind)
        inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(n,8))
        if next(iter(modules.values()))._expert_map is not None:
            incoming=torch.tensor(mapping,dtype=torch.int32,device=inputs[2].device)
            for name,m in modules.items():self.update(name,m,incoming)
        else:mapping=None
        return values,mapping

    @staticmethod
    def assert_alias(module):
        lookup,mapping=module._expert_lookup,module._expert_map
        assert lookup.untyped_storage().data_ptr()==mapping.untyped_storage().data_ptr()
        assert mapping.data_ptr()==lookup.data_ptr()+lookup.element_size()
        assert mapping.storage_offset()==1 and mapping.is_contiguous()
        assert mapping.dtype==lookup.dtype==torch.int32
        assert set(dict(module.named_buffers()))=={'_expert_lookup'}
        assert not module.state_dict()

    def lifecycle(self):
        def run():
            if self.t is None:return {'status':'skipped','reason':'PyTorch unavailable'}
            checks=[]
            def test(name,fn):
                try:checks.append(dict(name=name,status='passed',**(fn() or {})))
                except Exception as exc:checks.append({'name':name,'status':'failed','error':repr(exc),'traceback':traceback.format_exc(limit=6)})
            with torch.inference_mode():
                def correctness():
                    cases=0
                    for creation_mode in (False,True):
                        with torch.inference_mode(creation_mode):modules=self.modules()
                        for n in (9,32,128,256):
                            for dtype in ('int32','int64'):
                                inputs=self.inputs(n,dtype)
                                for kind in ('contiguous','permuted','shared'):
                                    for pattern in ('hot_local','sentinel','all_negative','all_remote'):
                                        values,mapping=self.load(modules,inputs,n,pattern,kind)
                                        for m in modules.values():assert not self.check_result(m(*inputs),m,inputs,values,mapping)
                                        cases+=1
                    return {'input_cases':cases,'output_checks':cases*3}
                test('three_variant_output_equivalence',correctness)
                def alias_and_direct_write():
                    module=self.modules()['single_owner'];self.assert_alias(module)
                    inputs=self.inputs(32);values=routes(32,'sentinel');inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(32,8))
                    before=module._expert_lookup.data_ptr()
                    for kind in ('permuted','shared','all_remote'):
                        mapping=expert_mapping(kind)
                        module._expert_map.copy_(torch.tensor(mapping,dtype=torch.int32,device=inputs[2].device))
                        self.check_result(module(*inputs),module,inputs,values,mapping)
                        assert module._expert_lookup[0].item()==-1 and module._expert_lookup.data_ptr()==before
                    try:module._expert_map=module._expert_map.clone()
                    except (AttributeError,KeyError):pass
                    else:raise AssertionError('Map replacement unexpectedly accepted')
                    self.assert_alias(module)
                    return {'direct_mutations_without_refresh':3,'replacement_rejected':True,'map_storage_offset_elements':1,'map_byte_offset':4,
                        'note':'This changes buffer introspection: _expert_map is a property, not a named buffer. Full framework compatibility remains untested.'}
                test('alias_direct_mutation_and_replacement',alias_and_direct_write)
                def moves():
                    module=self.modules(device='cpu')['single_owner']
                    devices=['cpu','cuda:0','cpu','cuda:0'] if self.gpu else ['cpu']
                    mapping=expert_mapping('shared')
                    for device in devices:
                        module.to(device=device,dtype=torch.float16);self.assert_alias(module)
                        module.update_expert_map_(torch.tensor(mapping,dtype=torch.int32,device=device))
                        assert module._expert_map.tolist()==mapping
                        assert module._expert_lookup[0].item()==-1
                    return {'device_sequence':devices,'alias_preserved':True,'note':'Movements precede capture; relocation of an existing graph is not supported.'}
                test('device_moves_and_dtype_preserve_alias',moves)
                def invalid_updates():
                    module=self.modules()['single_owner'];before=module._expert_lookup.clone()
                    for bad in (module._expert_map[:10],module._expert_map.float()):
                        try:module.update_expert_map_(bad)
                        except ValueError:pass
                        else:raise AssertionError('Invalid update accepted')
                        assert torch.equal(before,module._expert_lookup)
                    return {'rejected_before_mutation':2}
                test('invalid_update_contract',invalid_updates)
                def checkpoint_and_bytes():
                    modules=self.modules();sizes={name:sum(b.numel()*b.element_size() for b in m.buffers()) for name,m in modules.items()}
                    assert sizes=={'baseline':1024,'two_buffer':2052,'single_owner':1028}
                    module=modules['single_owner'];self.assert_alias(module)
                    module.register_buffer('_probe_marker',torch.tensor([5.],device=module._expert_lookup.device))
                    saved=io.BytesIO();torch.save(module.state_dict(),saved);saved.seek(0)
                    state=torch.load(saved,weights_only=True,map_location=module._expert_lookup.device)
                    assert set(state)=={'_probe_marker'}
                    module._probe_marker.zero_();module.load_state_dict(state);assert module._probe_marker.item()==5
                    return {'registered_buffer_bytes':sizes,'checkpoint_marker_roundtrip':True,'note':'Only marker/state_dict behavior; no actual model checkpoint load.'}
                test('owned_storage_and_checkpoint_exclusion',checkpoint_and_bytes)
                def branch_checks():
                    modules=self.modules();count=0
                    for n in (1,8):
                        for strided in (False,True):
                            inputs=self.inputs(n,strided=strided);values,mapping=self.load(modules,inputs,n,'sentinel')
                            flags=[self.check_result(m(*inputs),m,inputs,values,mapping) for m in modules.values()]
                            assert len(set(flags))==1;count+=1
                    modules=self.modules(no_map=True);inputs=self.inputs(9);values,mapping=self.load(modules,inputs,9,'sentinel')
                    for m in modules.values():assert not self.check_result(m(*inputs),m,inputs,values,mapping)
                    return {'small_branch_cases':count,'no_map_cases':1,'note':'Fused branch values are uninitialized until an omitted kernel and are not inspected. New map pointer is offset by four bytes; fused-kernel behavior/performance still needs target validation.'}
                test('small_branch_and_no_map_contract',branch_checks)
                if self.gpu:
                    def direct_write_graph():
                        module=self.modules()['single_owner'];inputs=self.inputs(128)
                        values=routes(128,'sentinel');inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(128,8))
                        graph,output,_=self.capture(lambda:module(*inputs));before=module._expert_lookup.data_ptr()
                        for kind in ('shared','permuted','all_remote'):
                            mapping=expert_mapping(kind);module._expert_map.copy_(torch.tensor(mapping,dtype=torch.int32,device='cuda'))
                            graph.replay();self.check_result(output,module,inputs,values,mapping)
                            assert module._expert_lookup.data_ptr()==before
                        return {'direct_write_graph_checks':3,'lookup_refresh_calls':0}
                    test('captured_graph_observes_direct_map_writes',direct_write_graph)
                else:checks.append({'name':'direct_write_cuda_graph','status':'skipped','reason':'No CUDA GPU'})
            write_csv(self.out/'lifecycle_checks.csv',checks)
            return {'status':'failed' if any(c['status']=='failed' for c in checks) else 'passed','checks':checks,
                'limitations':'Source slices only. Property/buffer introspection compatibility, meta-device loading, full checkpoint loading and real fused alignment/GEMMs remain untested.'}
        return self.section('lifecycle',run)

    def make_case(self,n,k):
        with torch.inference_mode(False):modules=self.modules()
        inputs=self.inputs(n);values,mapping=self.load(modules,inputs,n)
        incoming=torch.tensor(mapping,dtype=torch.int32,device='cuda')
        captures={name:self.capture(lambda m=m:m(*inputs)) for name,m in modules.items()}
        def make_static(graph,output):
            def fn():
                for _ in range(k):graph.replay()
                return output
            return fn
        def make_external(name,module,static):
            def fn():
                self.update(name,module,incoming)
                return static()
            return fn
        variants={}
        for name,m in modules.items():
            static=make_static(captures[name][0],captures[name][1])
            variants[name+'_static']=static;variants[name+'_external']=make_external(name,m,static)
        return modules,inputs,incoming,captures,variants

    def check_case(self,bundle,n):
        modules,inputs,incoming,captures,variants=bundle
        count=0
        for kind,pattern in [('permuted','sentinel'),('shared','all_negative')]:
            values,mapping=self.load(modules,inputs,n,pattern,kind)
            incoming.copy_(torch.tensor(mapping,dtype=torch.int32))
            for name,fn in variants.items():
                module=modules[name.rsplit('_',1)[0]]
                self.check_result(fn(),module,inputs,values,mapping);count+=1
        values,mapping=self.load(modules,inputs,n);incoming.copy_(torch.tensor(mapping,dtype=torch.int32))
        return count,values,mapping

    def benchmark(self):
        def run():
            if not self.gpu or self.report['sections'].get('lifecycle',{}).get('status')!='passed':return {'status':'skipped','reason':'Requires passing CUDA lifecycle tests'}
            with torch.inference_mode():
                for trial in range(self.trials):
                    configs=[(n,k) for n in (32,128) for k in (1,2,4,8,64)];random.Random(3300+trial).shuffle(configs)
                    for index,(n,k) in enumerate(configs):
                        identity={'trial':trial,'case_order':index,'tokens':n,'dtype':'int64','pattern':'hot_local','scope':'ownership','replays_per_cycle':k};bundle=None
                        try:
                            bundle=self.make_case(n,k);checked,values,mapping=self.check_case(bundle,n)
                            modules,inputs,_,_,variants=bundle
                            self.measure(variants,identity)
                            for name,fn in variants.items():self.check_result(fn(),modules[name.rsplit('_',1)[0]],inputs,values,mapping)
                            self.cases.append(dict(identity,status='passed',changed_input_schedule_checks=checked))
                        except Exception as exc:
                            self.rows=[r for r in self.rows if not all(r[key]==value for key,value in identity.items())]
                            self.raw=[r for r in self.raw if not all(r[key]==value for key,value in identity.items())]
                            self.cases.append(dict(identity,status='failed',error=repr(exc),traceback=traceback.format_exc(limit=6)))
                        finally:torch.cuda.synchronize();bundle=None
                        if (index+1)%5==0:self.save();print('Ownership trial',trial+1,index+1,'/',len(configs),flush=True)
            failed=sum(c['status']!='passed' for c in self.cases)
            return {'status':'passed' if not failed else 'partial','cases':len(self.cases),'failed_cases':failed,'changed_input_schedule_checks':sum(c.get('changed_input_schedule_checks',0) for c in self.cases)}
        return self.section('benchmark',run)

    def profile(self):
        def run():
            if not self.gpu or self.report['sections'].get('lifecycle',{}).get('status')!='passed':return {'status':'skipped','reason':'Requires passing CUDA lifecycle tests'}
            with torch.inference_mode():
                bundle=self.make_case(128,4);self.check_case(bundle,128)
                chosen={n:f for n,f in bundle[-1].items() if n.endswith('_external')}
                for fn in chosen.values():
                    for _ in range(5):fn()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as prof:
                    for i in range(4):
                        names=list(chosen);random.Random(910+i).shuffle(names)
                        for name in names:
                            with torch.profiler.record_function('mapping/'+name):chosen[name]()
                        prof.step()
                    torch.cuda.synchronize()
                path=self.out/'ownership_trace.json';prof.export_chrome_trace(str(path))
                result=parse_trace(path,self.out)
                result['note']+=' Graph-node correlation may be unavailable; retain partial status and unmatched events instead of fabricating totals.'
                return result
        return self.section('profiler',run)

    def export(self):
        comparisons=[]
        for case in self.cases:
            if case['status']!='passed':continue
            key={k:case[k] for k in ('trial','tokens','dtype','pattern','scope','replays_per_cycle')}
            rows={r['variant']:r for r in self.rows if all(r[k]==v for k,v in key.items())}
            for scenario in ('static','external'):
                for candidate in ('two_buffer','single_owner'):
                    b,c=rows['baseline_'+scenario],rows[candidate+'_'+scenario];k=case['replays_per_cycle']
                    comparisons.append(dict(key,scenario=scenario,candidate=candidate,baseline_cycle_us=b['event_us_median'],candidate_cycle_us=c['event_us_median'],baseline_over_candidate=b['event_us_median']/c['event_us_median'],baseline_us_per_replay=b['event_us_median']/k,candidate_us_per_replay=c['event_us_median']/k,wall_saved_us=b['wall_us_median']-c['wall_us_median']))
        self.report['totals']={'cases':len(self.cases),'timing_rows':len(self.rows),'raw_samples':len(self.raw)}
        self.save();write_csv(self.out/'comparisons.csv',comparisons)
        lines=['NOTEBOOK 07: SINGLE-OWNER LOOKUP','Run: '+self.report['run_id'],'',self.report['scope'],'']
        for name in ('environment','lifecycle','benchmark','profiler'):lines.append(name+': '+self.report['sections'].get(name,{}).get('status','not_run'))
        lines+=['','EXTERNAL UPDATE: normalized us per replay, including update share','Trial Tokens K Candidate Baseline Candidate Ratio']
        for r in comparisons:
            if r['scenario']=='external':lines.append('{trial} {tokens} {replays_per_cycle} {candidate} {baseline_us_per_replay:.2f} {candidate_us_per_replay:.2f} {baseline_over_candidate:.3f}'.format(**r))
        lines+=['','Contract: '+self.report['contract'],
            'The two-buffer and single-owner variants use identical forward arithmetic. Single ownership removes explicit refresh; direct in-place map writes become visible through the same storage.',
            'The property creates a view on access, whose host overhead is included where executed. It changes named-buffer introspection and prevents map attribute replacement. These integration constraints are not waived by passing this harness.',
            'New map storage offset is one int32/four bytes. Small fused branch selection is checked, but actual fused-kernel correctness/performance with that pointer offset is NOT validated on T4.',
            'Map updates occur outside existing prefix graphs in external scenarios. Static controls have no updates. Every cycle has K separate graph launches; compare matching candidate/baseline/scenario/trial.',
            'All calls use inference_mode. Incoming maps are already on GPU; setup/capture/input generation are outside timing. Cycles contain source prefixes, not actual full-model graphs.',
            'Both drafts remain unapplied. Real model loading, alignment/GEMMs, framework-wide buffer introspection, startup and H200 performance remain untested.','']
        (self.out/'SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name!='checksums.json'}
        (self.out/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        archive=self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():z.write(p,p.name)
        print('\n'.join(lines));print('Download:',archive);return archive
