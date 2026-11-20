"""Correctness gate for the unchanged two-buffer draft; no latency benchmark."""
import copy,hashlib,io,json,traceback,uuid,zipfile
from datetime import datetime,timezone
from pathlib import Path
import torch
if 'SourceInputs' not in globals():
    from input_harness import SourceInputs
    from source_slices import BaselineSourceSlice,CandidateSourceSlice
    from timing_support import write_csv

TOKENS=(0,1,2,8,9,16,64,128,512)
PATTERNS=('local','remote','mixed','negative')

def partition_map(rank):
    values=[-1]*256
    values[rank*32:(rank+1)*32]=range(32)
    return values

def gate_routes(n,rank,pattern):
    start=rank*32;local=list(range(start,start+32));remote=[i for i in range(256) if i not in local]
    values=[]
    for token in range(n):
        if pattern=='local':row=[local[(token*7+i)%32] for i in range(8)]
        elif pattern=='remote':row=[remote[(token*7+i)%len(remote)] for i in range(8)]
        elif pattern=='mixed':row=[-1,-2,-2147483648,0,255,local[token%32],local[(token+31)%32],remote[token%len(remote)]]
        else:row=[-1,-2,-2147483648,-3,-7,-99,-1000,-2147483647]
        values.extend(row)
    return values

class GateStudy(SourceInputs):
    def __init__(self,manifest,helper_source,output_root=None,force_cpu=False):
        self.manifest,self.helper_source=manifest,helper_source
        self.t,self.gpu,self.force_cpu=None,False,force_cpu
        root=Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        self.out=root/('v4_two_buffer_gate_'+run_id);self.out.mkdir(parents=True,exist_ok=False)
        self.rows,self.raw,self.cases=[],[],[];self.matrix_rows=[];self.update_rows=[];self.move_rows=[]
        self.report={'schema_version':12,'run_id':run_id,'source_manifest':manifest,
            'scope':'Correctness and lifecycle gate of baseline and unchanged selected two-buffer source slices. No latency benchmark, actual fused alignment/GEMM or complete model execution.',
            'config':{'expert_partitions':8,'global_experts':256,'local_experts':32,'tokens':TOKENS,'patterns':PATTERNS,'id_dtypes':['int32','int64'],'layouts':['contiguous','strided'],'include_no_map':True},
            'selection':'Two-buffer is preferred only for static/sufficiently reused maps with explicit refresh after edits. Selected draft is unapplied.',
            'sections':{}}
        self.save()

    def modules(self,rank,no_map=False,device=None):
        device=device or ('cuda:0' if self.gpu else 'cpu')
        kwargs={'num_experts':256,'num_local_experts':256 if no_map else 32,'local_expert_offset':0 if no_map else rank*32}
        return {'baseline':BaselineSourceSlice(**kwargs).to(device),'two_buffer':CandidateSourceSlice(**kwargs).to(device)}

    def make_inputs(self,n,dtype='int32',strided=False,device=None):
        device=device or ('cuda:0' if self.gpu else 'cpu')
        ids=torch.empty((n,16 if strided else 8),dtype=getattr(torch,dtype),device=device)
        if strided:ids=ids[:,::2]
        return (torch.ones((n,32),dtype=torch.float16,device=device),torch.full((n,8),.125,dtype=torch.float32,device=device),ids)

    def fill_ids(self,inputs,values):inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(inputs[2].shape))

    def matrix(self):
        def run():
            outputs=deferred=0
            with torch.inference_mode():
                for rank in range(9):
                    no_map=rank==8;logical_rank=0 if no_map else rank
                    modules=self.modules(logical_rank,no_map=no_map);mapping=None if no_map else partition_map(rank)
                    for n in TOKENS:
                        for dtype in ('int32','int64'):
                            for strided in (False,True):
                                for pattern in PATTERNS:
                                    row={'rank':'no_map' if no_map else rank,'tokens':n,'id_dtype':dtype,'strided':strided,'pattern':pattern}
                                    try:
                                        inputs=self.make_inputs(n,dtype,strided);values=gate_routes(n,logical_rank,pattern);self.fill_ids(inputs,values)
                                        flags=[]
                                        for m in modules.values():
                                            result=m(*inputs);flag=self.check_result(result,m,inputs,values,mapping)
                                            if n:assert result[2].data_ptr()==inputs[1].data_ptr()
                                            flags.append(flag);outputs+=1;deferred+=int(flag)
                                        assert flags[0]==flags[1]
                                        row.update(status='passed',fused_deferred=flags[0])
                                    except Exception as exc:row.update(status='failed',error=repr(exc),traceback=traceback.format_exc(limit=5))
                                    self.matrix_rows.append(row)
                    print('Partition matrix',rank+1,'/ 9',flush=True)
            write_csv(self.out/'matrix_checks.csv',self.matrix_rows)
            failed=sum(r['status']!='passed' for r in self.matrix_rows)
            return {'status':'passed' if not failed else 'partial','input_cases':len(self.matrix_rows),'failed_cases':failed,'outputs_checked':outputs,
                'deferred_fused_metadata_outputs':deferred,'value_checked_outputs':outputs-deferred,
                'note':'Deferred fused output ID values are uninitialized and never inspected. Full actual fused-kernel correctness remains untested.'}
        return self.section('matrix',run)

    def update_gate(self):
        def run():
            if not self.gpu:return {'status':'skipped','reason':'Graph refresh checks need CUDA; CPU matrix/contracts still run'}
            count=0;pointer_checks=0
            with torch.inference_mode():
                for rank in range(8):
                    with torch.inference_mode(False):modules=self.modules(rank)
                    b,c=modules['baseline'],modules['two_buffer'];inputs=self.make_inputs(32)
                    values=list(range(256));values[:3]=[-1,-2,-2147483648];self.fill_ids(inputs,values)
                    captures={name:self.capture(lambda m=m:m(*inputs)) for name,m in modules.items()}
                    pointers=(b._expert_map.data_ptr(),c._expert_map.data_ptr(),c._expert_lookup.data_ptr())
                    reverse=partition_map(rank);reverse[rank*32:(rank+1)*32]=reversed(range(32))
                    maps=[('initial',partition_map(rank)),('all_remote',[-1]*256),('reversed',reverse),('next_partition',partition_map((rank+1)%8))]
                    for path in ('checked_update','direct_then_refresh'):
                        for label,mapping in maps:
                            incoming=torch.tensor(mapping,dtype=torch.int32,device='cuda')
                            b._expert_map.fill_(-7);c._expert_map.fill_(-7);c._expert_lookup[1:].fill_(-7)
                            b._expert_map.copy_(incoming)
                            if path=='checked_update':c.update_expert_map_(incoming)
                            else:c._expert_map.copy_(incoming);c.refresh_expert_lookup_()
                            assert c._expert_lookup[0].item()==-1
                            assert torch.equal(c._expert_map,c._expert_lookup[1:])
                            for name,m in modules.items():
                                graph,out,_=captures[name];graph.replay();self.check_result(out,m,inputs,values,mapping);count+=1
                            assert pointers==(b._expert_map.data_ptr(),c._expert_map.data_ptr(),c._expert_lookup.data_ptr());pointer_checks+=1
                            self.update_rows.append({'rank':rank,'path':path,'mapping':label,'status':'passed','poisoned_state_replaced':True})
                    print('Update gate',rank+1,'/ 8',flush=True)
            write_csv(self.out/'graph_update_checks.csv',self.update_rows)
            return {'status':'passed','update_cycles':len(self.update_rows),'graph_output_checks':count,'stable_pointer_checks':pointer_checks,
                'note':'All updates ordered on the current stream before graph replay. Cross-stream/concurrent mutations are outside this gate.'}
        return self.section('updates',run)

    def moves_gate(self):
        def run():
            if not self.gpu:return {'status':'skipped','reason':'Device-move and recapture checks need CUDA'}
            outputs=0
            with torch.inference_mode():
                for rank in range(8):
                    modules=self.modules(rank);values=gate_routes(32,rank,'mixed');mapping=partition_map(rank)
                    inputs=self.make_inputs(32);self.fill_ids(inputs,values)
                    # Validate an initial graph, then release it before relocating buffers.
                    for m in modules.values():
                        graph,out,_=self.capture(lambda m=m:m(*inputs));graph.replay();self.check_result(out,m,inputs,values,mapping);outputs+=1
                        del graph,out
                    cpu_inputs=self.make_inputs(32,device='cpu');self.fill_ids(cpu_inputs,values)
                    for m in modules.values():m.to('cpu');self.check_result(m(*cpu_inputs),m,cpu_inputs,values,mapping);outputs+=1
                    for m in modules.values():m.to(device='cuda',dtype=torch.float16)
                    for name,m in modules.items():
                        assert m._expert_map.dtype==torch.int32 and m._expert_map.storage_offset()==0
                        if name=='two_buffer':assert torch.equal(m._expert_map,m._expert_lookup[1:])
                        graph,out,_=self.capture(lambda m=m:m(*inputs));graph.replay();self.check_result(out,m,inputs,values,mapping);outputs+=1
                        del graph,out
                    self.move_rows.append({'rank':rank,'sequence':'cuda -> cpu -> cuda -> fresh capture','status':'passed','output_checks':6})
            write_csv(self.out/'move_checks.csv',self.move_rows)
            return {'status':'passed','partitions':8,'output_checks':outputs,'note':'Old graphs are released before moves. No stale graph is replayed after relocation; recapture is explicit.'}
        return self.section('moves',run)

    def contracts_gate(self):
        def run():
            rejected=clone_checks=0
            with torch.inference_mode():
                for rank in range(8):
                    c=self.modules(rank)['two_buffer'];before=(c._expert_map.clone(),c._expert_lookup.clone())
                    for bad in (c._expert_map[:255],c._expert_map.float(),torch.empty(256,dtype=torch.int32,device='meta')):
                        try:c.update_expert_map_(bad)
                        except ValueError:rejected+=1
                        else:raise AssertionError('Malformed update accepted')
                        assert torch.equal(before[0],c._expert_map) and torch.equal(before[1],c._expert_lookup)
                    assert set(dict(c.named_buffers()))=={'_expert_map','_expert_lookup'} and not c.state_dict()
                    clone=copy.deepcopy(c)
                    assert clone._expert_map.data_ptr()!=c._expert_map.data_ptr() and clone._expert_lookup.data_ptr()!=c._expert_lookup.data_ptr()
                    assert torch.equal(clone._expert_map,clone._expert_lookup[1:]);clone_checks+=1
                c=self.modules(0,no_map=True)['two_buffer'];c.refresh_expert_lookup_()
                assert c._expert_map is None and c._expert_lookup is None
                try:c.update_expert_map_(torch.zeros(256,dtype=torch.int32,device='cuda' if self.gpu else 'cpu'))
                except ValueError:rejected+=1
                else:raise AssertionError('Enabled mapping without module reconstruction')
                c=self.modules(0)['two_buffer'];c.register_buffer('_marker',torch.tensor([5.],device=c._expert_map.device))
                stream=io.BytesIO();torch.save(c.state_dict(),stream);stream.seek(0)
                state=torch.load(stream,weights_only=True,map_location=c._expert_map.device)
                assert set(state)=={'_marker'};c._marker.zero_();c.load_state_dict(state);assert c._marker.item()==5
            return {'status':'passed','invalid_updates_rejected_before_mutation':rejected,'independent_consistent_deepcopies':clone_checks,
                'nonpersistent_buffer_exclusion':True,'marker_checkpoint_roundtrip':True,
                'note':'Marker/sliced-module state_dict checks only, not actual model checkpoint or meta-device loading. Meta tensor is only an invalid update input.'}
        return self.section('contracts',run)

    def export(self):
        # Persist any rows completed before a caught section failure.
        for name,rows in [('matrix_checks.csv',self.matrix_rows),('graph_update_checks.csv',self.update_rows),('move_checks.csv',self.move_rows)]:write_csv(self.out/name,rows)
        sections=('matrix','updates','moves','contracts')
        self.report['gate_status']='passed' if self.gpu and all(self.report['sections'].get(s,{}).get('status')=='passed' for s in sections) else 'incomplete_or_failed'
        self.report['remaining_integration_gates']=['Actual fused small-input alignment and GEMMs','Full serving/checkpoint/buffer integration','All production map writers refresh before consumption','Recapture after buffer replacement/movement','Concurrent-stream ordering','H200/full-serving performance']
        self.save()
        lines=['NOTEBOOK 12: TWO-BUFFER CORRECTNESS GATE','Run: '+self.report['run_id'],'',self.report['scope'],'']
        for name in ('environment',)+sections:lines.append(name+': '+self.report['sections'].get(name,{}).get('status','not_run'))
        lines+=['','Gate status: '+self.report['gate_status'],
            'Selected draft is the exact earlier two-buffer candidate, still UNAPPLIED. No new algorithm or active V4 changes.',
            'Any direct map edit requires refresh before consumers. Checked updates perform that refresh. Updates must be ordered before replay; replacing/moving buffers requires recapture.',
            'Fused-branch local ID values are not inspected before the omitted kernel. Matrix counts distinguish value checks from metadata-only checks.',
            'There are no latency measurements in this notebook. Section wall times are execution diagnostics, not performance evidence.',
            'Passing this source-slice gate is not approval to deploy or proof of complete-model correctness. Remaining integration gates:']+self.report['remaining_integration_gates']+['']
        (self.out/'SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name!='checksums.json'}
        (self.out/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        archive=self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():z.write(p,p.name)
        print('\n'.join(lines));print('Download:',archive);return archive
