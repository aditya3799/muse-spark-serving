"""Mapping ownership prototype, not the full serving module or an applied patch."""
import copy
import hashlib
import io
import json
import random
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import torch

if 'ExperimentCore' not in globals():
    from timing_support import ExperimentCore, routes, reference, expert_mapping, write_csv, parse_trace


class BaselineMappingModule(torch.nn.Module):
    """Mirror nonpersistent map ownership and the exact serving fallback helper."""
    def __init__(self, mapping, helper):
        super().__init__()
        self._baseline_helper = helper
        self.register_buffer('_expert_map', None if mapping is None else mapping.clone(), persistent=False)

    def _validate_update(self, mapping):
        if self._expert_map is None:
            raise ValueError('Cannot enable expert mapping in place; reconstruct module and recapture graphs')
        if mapping.shape != self._expert_map.shape or mapping.dtype != torch.int32 or mapping.device != self._expert_map.device:
            raise ValueError('Update must preserve shape, int32 dtype and device')

    @torch.no_grad()
    def update_expert_map_(self, mapping):
        self._validate_update(mapping)
        self._expert_map.copy_(mapping)

    def forward(self, topk_ids):
        local = topk_ids.to(torch.int32)
        if self._expert_map is not None:
            local = self._baseline_helper(local, self._expert_map)
        return local.contiguous()


class CachedMappingModule(BaselineMappingModule):
    """Experimental companion buffer with an explicit update/refresh contract."""
    def __init__(self, mapping, helper):
        super().__init__(mapping, helper)
        lookup = None if self._expert_map is None else torch.cat((self._expert_map.new_full((1,), -1), self._expert_map))
        self.register_buffer('_expert_lookup', lookup, persistent=False)

    @torch.no_grad()
    def refresh_expert_lookup_(self):
        if self._expert_map is not None:
            self._expert_lookup[0].fill_(-1)
            self._expert_lookup[1:].copy_(self._expert_map)

    @torch.no_grad()
    def update_expert_map_(self, mapping):
        self._validate_update(mapping)
        self._expert_map.copy_(mapping)
        self.refresh_expert_lookup_()

    def forward(self, topk_ids):
        local = topk_ids.to(torch.int32)
        if self._expert_lookup is not None:
            local = self._expert_lookup[local.clamp_min(-1) + 1]
        return local.contiguous()


class IntegrationStudy(ExperimentCore):
    def __init__(self, manifest, helper_source, output_root=None, force_cpu=False,
                 samples=11, repeats=20, trials=3, tokens=(9,32,128,256)):
        assert hashlib.sha256(helper_source.encode()).hexdigest()==manifest['mapping_helper_sha256']
        self.manifest, self.helper_source = manifest, helper_source
        self.t, self.gpu, self.force_cpu = None, False, force_cpu
        self.samples, self.repeats, self.trials, self.tokens = samples, repeats, trials, tuple(tokens)
        root=Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        self.out=root/('v4_mapping_integration_'+run_id)
        self.out.mkdir(parents=True,exist_ok=False)
        self.rows,self.raw,self.cases=[],[],[]
        self.report={'schema_version':4,'run_id':run_id,'source_manifest':manifest,
            'scope':'Isolated nn.Module ownership/lifecycle prototype around the exact mapping fallback. NOT the complete MuseSpark class, real alignment, a model run or an applied serving patch.',
            'config':{'tokens':list(tokens),'trials':trials,'samples':samples,'repeats':repeats,'dtypes':['int32','int64'],'patterns':['uniform','hot_local']},
            'cache_contract':'Use update_expert_map_ for changes, or explicitly refresh after direct edits. No automatic stale-cache detection. Updates and graph replay must be ordered on the same stream or by events. Recreate CUDA graphs after moving/replacing buffers.',
            'sections':{}}
        (self.out/'mapping_helper.py').write_text(helper_source,encoding='utf-8')
        self.save()

    def pair(self, mapping, device=None):
        mapping=None if mapping is None else torch.tensor(mapping,dtype=torch.int32)
        baseline=BaselineMappingModule(mapping,self.baseline)
        candidate=CachedMappingModule(mapping,self.baseline)
        device=device or ('cuda:0' if self.gpu else 'cpu')
        return baseline.to(device),candidate.to(device)

    def check_pair(self, pair, values, mapping, tokens, dtype=torch.int64, strided=False):
        device=next(pair[0].buffers(),torch.empty(0)).device if mapping is not None else torch.device('cuda:0' if self.gpu else 'cpu')
        ids=torch.tensor(values,dtype=dtype,device=device).reshape(tokens,8)
        if strided:
            backing=torch.empty((tokens,16),dtype=dtype,device=device)
            backing[:,::2].copy_(ids)
            ids=backing[:,::2]
        expected=values if mapping is None else reference(values,mapping)
        for module in pair:
            got=module(ids)
            assert got.dtype==torch.int32 and got.shape==ids.shape and got.device==ids.device and got.is_contiguous()
            assert got.flatten().tolist()==expected
        return ids

    def lifecycle(self):
        def run():
            if self.t is None:
                return {'status':'skipped','reason':'PyTorch unavailable'}
            checks=[]
            def test(name,fn):
                try:
                    details=fn() or {}
                    checks.append(dict(name=name,status='passed',**details))
                except Exception as exc:
                    checks.append({'name':name,'status':'failed','error':repr(exc),'traceback':traceback.format_exc(limit=5)})
            def correctness():
                count=0
                for kind in ('contiguous','permuted','shared','all_remote'):
                    mapping=expert_mapping(kind)
                    pair=self.pair(mapping)
                    for n in (0,1,8,9,32,128,256):
                        for dtype in (torch.int32,torch.int64):
                            for strided in (False,True):
                                for pattern in ('uniform','hot_local','all_remote','all_local','sentinel','all_negative'):
                                    self.check_pair(pair,routes(n,pattern),mapping,n,dtype,strided)
                                    count+=1
                return {'input_cases':count,'module_output_checks':count*2}
            test('module_output_equivalence',correctness)
            def no_mapping():
                pair=self.pair(None)
                for n in (0,1,8,9,128):
                    self.check_pair(pair,routes(n,'sentinel'),None,n)
                assert pair[1]._expert_lookup is None
                return {'cases':5,'note':'No-map path preserves the int32 cast and does not normalize negative IDs.'}
            test('no_expert_map_passthrough',no_mapping)
            def invalid_update():
                pair=self.pair(expert_mapping())
                for module in pair:
                    before=module._expert_map.clone()
                    for bad in (before[:10],before.float()):
                        try: module.update_expert_map_(bad)
                        except ValueError: pass
                        else: raise AssertionError('Invalid update accepted')
                        assert torch.equal(before,module._expert_map)
                return {'rejected_updates':4}
            test('invalid_update_rejected_before_mutation',invalid_update)
            def stale_cache():
                mapping=expert_mapping()
                pair=self.pair(mapping)
                values=[64]*72
                ids=self.check_pair(pair,values,mapping,9)
                pair[1]._expert_map[64]=31
                changed=list(mapping);changed[64]=31
                stale=pair[1](ids).flatten().tolist()!=reference(values,changed)
                assert stale, 'Expected stale-cache hazard was not demonstrated'
                pair[1].refresh_expert_lookup_()
                assert pair[1](ids).flatten().tolist()==reference(values,changed)
                return {'stale_result_observed_before_refresh':True,'correct_after_refresh':True,
                        'note':'This is an expected-hazard test, not automatic detection in forward. Direct mutations must be followed by refresh.'}
            test('direct_mutation_hazard_and_explicit_refresh',stale_cache)
            def updates():
                pair=self.pair(expert_mapping())
                pointers=[(m._expert_map.data_ptr(),getattr(m,'_expert_lookup',m._expert_map).data_ptr()) for m in pair]
                for kind in ('permuted','shared','all_remote','contiguous'):
                    mapping=expert_mapping(kind)
                    for m in pair:
                        m.update_expert_map_(torch.tensor(mapping,dtype=torch.int32,device=m._expert_map.device))
                    self.check_pair(pair,routes(9,'sentinel'),mapping,9)
                assert pointers==[(m._expert_map.data_ptr(),getattr(m,'_expert_lookup',m._expert_map).data_ptr()) for m in pair]
                return {'mapping_updates':4,'buffer_addresses_preserved':True}
            test('update_api_and_storage_stability',updates)
            def movement():
                mapping=expert_mapping('shared')
                pair=self.pair(mapping,device='cpu')
                devices=['cpu','cuda:0','cpu','cuda:0'] if self.gpu else ['cpu']
                for device in devices:
                    for m in pair: m.to(device=device,dtype=torch.float16)
                    assert all(b.dtype==torch.int32 for m in pair for b in m.buffers())
                    self.check_pair(pair,routes(9,'sentinel'),mapping,9)
                return {'device_sequence':devices,'integer_buffer_dtypes_preserved':True,'note':'Movements occur before graph capture; graphs must be recaptured after relocation.'}
            test('module_device_movement_and_dtype',movement)
            def checkpoints():
                mapping=expert_mapping('permuted')
                pair=self.pair(mapping)
                for m in pair:
                    m.register_buffer('_probe_checkpoint_marker',torch.tensor([7.0],device=m._expert_map.device))
                    state=m.state_dict()
                    assert set(state)=={'_probe_checkpoint_marker'}
                    stream=io.BytesIO();torch.save(state,stream);stream.seek(0)
                    restored=torch.load(stream,weights_only=True,map_location=m._expert_map.device)
                    m._probe_checkpoint_marker.zero_();m.load_state_dict(restored)
                    assert m._probe_checkpoint_marker.item()==7.0
                self.check_pair(pair,routes(9,'sentinel'),mapping,9)
                return {'nonpersistent_mapping_buffers_excluded':True,'marker_roundtrip':True,'note':'Only prototype state_dict semantics, not a model checkpoint load.'}
            test('state_dict_nonpersistent_buffers',checkpoints)
            if self.gpu:
                def graph_updates():
                    pair=self.pair(expert_mapping())
                    ids=torch.tensor(routes(9,'uniform'),device='cuda',dtype=torch.int64).reshape(9,8)
                    captures=[self.capture(lambda m=m:m(ids)) for m in pair]
                    checked=0
                    for kind,pattern in [('permuted','sentinel'),('shared','hot_local'),('all_remote','all_local'),('contiguous','all_negative')]:
                        mapping=expert_mapping(kind);values=routes(9,pattern)
                        ids.copy_(torch.tensor(values,dtype=ids.dtype).reshape(9,8))
                        new=torch.tensor(mapping,dtype=torch.int32,device='cuda')
                        for m,(graph,output,_) in zip(pair,captures):
                            m.update_expert_map_(new)
                            graph.replay()
                            assert output.flatten().tolist()==reference(values,mapping)
                            checked+=1
                    return {'changed_input_graph_checks':checked}
                test('graph_replay_after_update_api',graph_updates)
                def stream_update():
                    pair=self.pair(expert_mapping())
                    ids=torch.tensor(routes(9,'hot_local'),device='cuda',dtype=torch.int64).reshape(9,8)
                    captures=[self.capture(lambda m=m:m(ids)) for m in pair]
                    mapping=expert_mapping('permuted',808)
                    new=torch.tensor(mapping,dtype=torch.int32,device='cuda')
                    producer=torch.cuda.Stream();producer.wait_stream(torch.cuda.current_stream())
                    ready=torch.cuda.Event()
                    with torch.cuda.stream(producer):
                        for m in pair: m.update_expert_map_(new)
                        ready.record()
                    torch.cuda.current_stream().wait_event(ready)
                    for graph,output,_ in captures:
                        graph.replay()
                        assert output.flatten().tolist()==reference(routes(9,'hot_local'),mapping)
                    return {'producer_stream_updates':2,'consumer_wait_event_verified':True}
                test('cross_stream_update_with_explicit_event',stream_update)
            else:
                checks.append({'name':'cuda_lifecycle_tests','status':'skipped','reason':'No active GPU'})
            write_csv(self.out/'lifecycle_checks.csv',checks)
            return {'status':'failed' if any(c['status']=='failed' for c in checks) else 'passed',
                    'checks':checks,'limitations':'Direct writes need explicit refresh; no automatic version guard. Shape/device replacement and device migration require graph recapture. Full model/meta-device loading and real aligner behavior are untested.'}
        return self.section('lifecycle',run)

    def benchmark(self):
        def run():
            if not self.gpu or self.report['sections'].get('lifecycle',{}).get('status')!='passed':
                return {'status':'skipped','reason':'Requires passing GPU lifecycle tests'}
            configs=[(n,d,p) for n in self.tokens for d in ('int32','int64') for p in ('uniform','hot_local')]
            for trial in range(self.trials):
                order=list(configs);random.Random(1400+trial).shuffle(order)
                for index,(n,dtype,pattern) in enumerate(order):
                    identity={'trial':trial,'case_order':index,'tokens':n,'dtype':dtype,'pattern':pattern,'scope':'module_fallback'}
                    captures={};variants={}
                    try:
                        mapping=expert_mapping();pair=self.pair(mapping)
                        ids=torch.tensor(routes(n,pattern),device='cuda',dtype=getattr(torch,dtype)).reshape(n,8)
                        fns={name:(lambda m=m:m(ids)) for name,m in zip(('baseline_module','cached_module'),pair)}
                        captures={name:self.capture(fn) for name,fn in fns.items()}
                        for kind,changed_pattern in [('permuted','sentinel'),('shared','hot_local'),('all_remote','all_local')]:
                            mapping=expert_mapping(kind);values=routes(n,changed_pattern)
                            ids.copy_(torch.tensor(values,dtype=ids.dtype).reshape(n,8))
                            for m in pair:m.update_expert_map_(torch.tensor(mapping,dtype=torch.int32,device='cuda'))
                            for graph,output,_ in captures.values():
                                graph.replay();self.check(output,reference(values,mapping),n)
                        mapping=expert_mapping();values=routes(n,pattern)
                        ids.copy_(torch.tensor(values,dtype=ids.dtype).reshape(n,8))
                        for m in pair:m.update_expert_map_(torch.tensor(mapping,dtype=torch.int32,device='cuda'))
                        for name,fn in fns.items():
                            graph,output,_=captures[name]
                            graph.replay();self.check(output,reference(values,mapping),n)
                            variants[name+'_eager']=fn;variants[name+'_graph']=graph.replay
                        self.measure(variants,identity)
                        for name,fn in fns.items():
                            self.check(fn(),reference(values,mapping),n)
                            graph,output,_=captures[name]
                            graph.replay();self.check(output,reference(values,mapping),n)
                        self.cases.append(dict(identity,status='passed',changed_input_graph_checks=6))
                    except Exception as exc:
                        self.rows=[r for r in self.rows if not all(r[k]==v for k,v in identity.items())]
                        self.raw=[r for r in self.raw if not all(r[k]==v for k,v in identity.items())]
                        self.cases.append(dict(identity,status='failed',error=repr(exc),traceback=traceback.format_exc(limit=6)))
                    finally:
                        torch.cuda.synchronize();variants.clear();captures.clear()
                    if (index+1)%8==0:
                        self.save();print('Trial',trial+1,'completed',index+1,'/',len(order),'module cases',flush=True)
                self.save()
            failed=sum(c['status']!='passed' for c in self.cases)
            return {'status':'passed' if not failed else 'partial','cases':len(self.cases),'failed_cases':failed,
                    'changed_input_graph_checks':sum(c.get('changed_input_graph_checks',0) for c in self.cases)}
        return self.section('benchmark',run)

    def update_cost(self):
        def run():
            if not self.gpu or self.report['sections'].get('lifecycle',{}).get('status')!='passed':
                return {'status':'skipped','reason':'Requires GPU lifecycle tests'}
            pair=self.pair(expert_mapping())
            mapping=expert_mapping('shared');new=torch.tensor(mapping,dtype=torch.int32,device='cuda')
            fns={name:(lambda m=m:m.update_expert_map_(new)) for name,m in zip(('baseline_update','cached_update'),pair)}
            captures={name:self.capture(fn) for name,fn in fns.items()}
            variants={}
            for name,fn in fns.items():variants[name+'_eager']=fn;variants[name+'_graph']=captures[name][0].replay
            for trial in range(self.trials):
                self.measure(variants,{'trial':trial,'case_order':-1,'tokens':0,'dtype':'int32','pattern':'shared','scope':'update_api'})
            self.check_pair(pair,routes(9,'hot_local'),mapping,9)
            return {'status':'passed','note':'Updates are timed separately: baseline copies map; candidate copies map and refreshes existing lookup. Forward timings assume mappings are reused. Do not sum separate medians into exact end-to-end latency.'}
        return self.section('update_cost',run)

    def profile(self):
        def run():
            if not self.gpu or self.report['sections'].get('lifecycle',{}).get('status')!='passed':
                return {'status':'skipped','reason':'Requires GPU lifecycle tests'}
            mapping=expert_mapping();pair=self.pair(mapping)
            values=routes(128,'hot_local');ids=torch.tensor(values,dtype=torch.int64,device='cuda').reshape(128,8)
            fns={name:(lambda m=m:m(ids)) for name,m in zip(('baseline_module','cached_module'),pair)}
            for fn in fns.values():
                for _ in range(10):fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,with_stack=False) as prof:
                for i in range(6):
                    names=list(fns);random.Random(60+i).shuffle(names)
                    for name in names:
                        with torch.profiler.record_function('mapping/'+name):output=fns[name]()
                    prof.step()
                torch.cuda.synchronize()
            self.check(output,reference(values,mapping),128)
            path=self.out/'mapping_trace.json';prof.export_chrome_trace(str(path))
            return parse_trace(path,self.out)
        return self.section('profiler',run)

    def export(self):
        self.report['totals']={'timing_rows':len(self.rows),'raw_samples':len(self.raw)}
        self.save()
        comparisons=[]
        for case in self.cases:
            if case['status']!='passed':continue
            key={k:case[k] for k in ('trial','tokens','dtype','pattern','scope')}
            rows={r['variant']:r for r in self.rows if all(r[k]==v for k,v in key.items())}
            for mode in ('eager','graph'):
                baseline,candidate=rows['baseline_module_'+mode],rows['cached_module_'+mode]
                comparisons.append(dict(key,execution=mode,baseline_us=baseline['event_us_median'],candidate_us=candidate['event_us_median'],baseline_over_candidate=baseline['event_us_median']/candidate['event_us_median'],wall_saved_us=baseline['wall_us_median']-candidate['wall_us_median']))
        write_csv(self.out/'comparisons.csv',comparisons)
        lines=['NOTEBOOK 04: MAPPING MODULE LIFECYCLE','Run: '+self.report['run_id'],'',self.report['scope'],'']
        for name in ('environment','lifecycle','benchmark','update_cost','profiler'):
            lines.append(name+': '+self.report['sections'].get(name,{}).get('status','not_run'))
        lines+=['','INT64 HOT-LOCAL MODULE FALLBACK: event medians, microseconds','Trial Tokens Mode Baseline Candidate Ratio']
        for r in comparisons:
            if r['dtype']=='int64' and r['pattern']=='hot_local':lines.append('{trial} {tokens} {execution} {baseline_us:.2f} {candidate_us:.2f} {baseline_over_candidate:.3f}'.format(**r))
        lines+=['','Lifecycle contract: '+self.report['cache_contract'],
            'The stale-cache test intentionally observes incorrect output after a direct edit, then verifies explicit refresh. A passing test does not mean direct edits are safe or automatically detected.',
            'Both mapping buffers are nonpersistent; reconstruction must supply correct rank/mapping metadata. The checkpoint test uses only a marker tensor, not real model weights.',
            'The <=64-route fused alignment path is not executed or altered. Empty/small inputs only test the mapping wrapper. No model, real aligner, multi-GPU communication, scheduler/tokenizer startup or H200 performance is measured.',
            'Update costs are separate from forward timing. CUDA-event intervals may contain launch gaps; graph replay changes dispatch and allocator behavior.',
            'Timing is repeated/interleaved in three trials with no profiler. Trace is separate and excludes annotation spans from device-operation counts.',
            'No source patch is applied or automatically promoted. Return the ZIP for integration review and the next experiment.','']
        (self.out/'SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name!='checksums.json'}
        (self.out/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        archive=self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():z.write(p,p.name)
        print('\n'.join(lines));print('Download:',archive)
        return archive
