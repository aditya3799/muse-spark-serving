"""Event-ordered producer updates and consumer graph replay; correctness only."""
import hashlib,json,uuid,zipfile
from datetime import datetime,timezone
from pathlib import Path
import torch
if 'GateStudy' not in globals():
    from gate_study import GateStudy,partition_map
    from timing_support import write_csv

class StreamStudy(GateStudy):
    def __init__(self,manifest,helper_source,output_root=None,force_cpu=False):
        self.manifest,self.helper_source=manifest,helper_source
        self.t,self.gpu,self.force_cpu=None,False,force_cpu
        root=Path(output_root) if output_root else (Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd())
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
        self.out=root/('v4_stream_ordered_'+run_id);self.out.mkdir(parents=True,exist_ok=False)
        self.rows,self.raw,self.cases=[],[],[];self.handoff_rows=[];self.smoke_rows=[]
        self.report={'schema_version':13,'run_id':run_id,'source_manifest':manifest,
            'scope':'Correctness of explicit CUDA event ordering for unchanged two-buffer source prefixes. No deliberately unsynchronized race, latency benchmark, actual fused kernel or full serving execution.',
            'config':{'partitions':8,'phases':['initial_capture','recapture_after_move'],'consumer_streams':['default','dedicated'],
                'update_paths':['checked_update','direct_then_refresh'],'map_states':['initial','all_remote','reversed','next_partition'],'rounds':3,'tokens':32,'id_dtype':'int32','score_dtype':'float32'},
            'contract':'Producer waits for prior completion before overwriting shared buffers, updates IDs/map/lookup, and records ready. Consumer waits ready, replays one combined baseline/candidate graph, and records done. Host waits done before reading outputs. Only device 0 used.',
            'sections':{}}
        self.save()

    def smoke(self):
        def run():
            with torch.inference_mode():
                for rank in range(8):
                    modules=self.modules(rank);inputs=self.make_inputs(32)
                    values=list(range(256));values[:3]=[-1,-2,-2147483648];self.fill_ids(inputs,values)
                    mapping=partition_map(rank)
                    for name,m in modules.items():
                        assert not self.check_result(m(*inputs),m,inputs,values,mapping)
                        self.smoke_rows.append({'rank':rank,'variant':name,'status':'passed'})
            write_csv(self.out/'smoke_checks.csv',self.smoke_rows)
            return {'status':'passed','output_checks':len(self.smoke_rows)}
        return self.section('smoke',run)

    def handoff(self):
        def run():
            if not self.gpu:return {'status':'skipped','reason':'CUDA events and graph replay require GPU'}
            if any(self.report['sections'].get(s,{}).get('status')!='passed' for s in ('smoke','contracts')):
                return {'status':'skipped','reason':'Requires passing smoke and update-contract gates'}
            outputs=pointer_checks=recaptures=0
            with torch.inference_mode():
                for rank in range(8):
                    with torch.inference_mode(False):modules=self.modules(rank)
                    b,c=modules['baseline'],modules['two_buffer'];inputs=self.make_inputs(32)
                    self.fill_ids(inputs,list(range(256)))
                    reverse=partition_map(rank);reverse[rank*32:(rank+1)*32]=reversed(range(32))
                    maps=[('initial',partition_map(rank)),('all_remote',[-1]*256),('reversed',reverse),('next_partition',partition_map((rank+1)%8))]
                    # All input data already resides on GPU before the handoff loops.
                    incoming_maps={name:torch.tensor(mapping,dtype=torch.int32,device='cuda:0') for name,mapping in maps}
                    ids_by_round={}
                    for cycle_round in range(3):
                        shift=(rank*5+cycle_round*17)%256;values=list(range(256));values=values[shift:]+values[:shift]
                        values[:3]=[-1,-2,-2147483648]
                        ids_by_round[cycle_round]=(values,torch.tensor(values,dtype=torch.int32,device='cuda:0').reshape(32,8))
                    for phase in ('initial_capture','recapture_after_move'):
                        if phase=='recapture_after_move':
                            # Previous graph was released after all completion events finished.
                            torch.cuda.synchronize()
                            for m in modules.values():m.to('cpu')
                            for m in modules.values():m.to('cuda:0')
                            recaptures+=1
                        graph,out,_=self.capture(lambda:tuple(m(*inputs) for m in modules.values()))
                        producer=torch.cuda.Stream(device=0);dedicated=torch.cuda.Stream(device=0)
                        default=torch.cuda.default_stream(0)
                        producer.wait_stream(torch.cuda.current_stream())
                        pointers=(b._expert_map.data_ptr(),c._expert_map.data_ptr(),c._expert_lookup.data_ptr(),inputs[2].data_ptr())
                        for consumer_name,consumer in [('default',default),('dedicated',dedicated)]:
                            assert producer.cuda_stream!=consumer.cuda_stream
                            ready,done=torch.cuda.Event(),torch.cuda.Event();has_previous=False
                            for path in ('checked_update','direct_then_refresh'):
                                for cycle_round in range(3):
                                    values,incoming_ids=ids_by_round[cycle_round]
                                    for map_name,mapping in maps:
                                        with torch.cuda.stream(producer):
                                            if has_previous:producer.wait_event(done)
                                            # Poison first so skipped updates are observable; all writes ordered here.
                                            b._expert_map.fill_(-7);c._expert_map.fill_(-7);c._expert_lookup[1:].fill_(-7)
                                            inputs[2].copy_(incoming_ids)
                                            b._expert_map.copy_(incoming_maps[map_name])
                                            if path=='checked_update':c.update_expert_map_(incoming_maps[map_name])
                                            else:c._expert_map.copy_(incoming_maps[map_name]);c.refresh_expert_lookup_()
                                            ready.record(producer)
                                        with torch.cuda.stream(consumer):
                                            consumer.wait_event(ready)
                                            graph.replay()
                                            done.record(consumer)
                                        # Event-scoped wait: no device-wide synchronization in the handoff.
                                        done.synchronize();has_previous=True
                                        for output,m in zip(out,modules.values()):
                                            assert not self.check_result(output,m,inputs,values,mapping);outputs+=1
                                        assert c._expert_lookup[0].item()==-1
                                        assert torch.equal(c._expert_map,c._expert_lookup[1:])
                                        assert pointers==(b._expert_map.data_ptr(),c._expert_map.data_ptr(),c._expert_lookup.data_ptr(),inputs[2].data_ptr());pointer_checks+=1
                                        self.handoff_rows.append({'rank':rank,'phase':phase,'consumer':consumer_name,'update_path':path,'round':cycle_round,'mapping':map_name,
                                            'producer_stream_handle':int(producer.cuda_stream),'consumer_stream_handle':int(consumer.cuda_stream),
                                            'status':'passed','output_checks':2,'ready_wait_used':True,'completion_event_wait_used':True,'stable_pointers':True})
                        del graph,out
                        # All consumers completed before buffers or streams leave this phase.
                        torch.cuda.synchronize()
                    print('Event-ordered partitions',rank+1,'/ 8',flush=True)
                    self.save()
            write_csv(self.out/'stream_handoff_checks.csv',self.handoff_rows)
            return {'status':'passed','handoff_cycles':len(self.handoff_rows),'output_checks':outputs,'stable_pointer_checks':pointer_checks,'partitions_recaptured_after_move':recaptures,
                'device_wide_sync_in_handoff':False,'device_wide_sync_during_setup_teardown':True,'intentionally_unsynchronized_cases':0,
                'note':'Host checks wait for each completion event, so this validates event-ordered handoff and reuse, not overlapped multi-request throughput or arbitrary concurrent writers.'}
        return self.section('handoff',run)

    def export(self):
        write_csv(self.out/'smoke_checks.csv',self.smoke_rows);write_csv(self.out/'stream_handoff_checks.csv',self.handoff_rows)
        self.report['gate_status']='passed' if self.gpu and all(self.report['sections'].get(s,{}).get('status')=='passed' for s in ('smoke','contracts','handoff')) else 'incomplete_or_failed'
        self.report['remaining_gates']=['Actual fused alignment and expert GEMMs','Full serving/checkpoint/buffer integration','Verify production refresh and event ordering call sites','Actual runtime router output layouts/backend','H200/full-serving performance']
        self.save()
        lines=['NOTEBOOK 13: EXPLICIT STREAM ORDERING','Run: '+self.report['run_id'],'',self.report['scope'],'']
        for name in ('environment','smoke','contracts','handoff'):lines.append(name+': '+self.report['sections'].get(name,{}).get('status','not_run'))
        lines+=['','Gate status: '+self.report['gate_status'],self.report['contract'],
            'All incoming map/ID tensors are allocated before handoff loops and remain alive. Completion events finish before host validation or buffer reuse. Setup/teardown may use device-wide synchronization; handoff uses event-scoped ordering.',
            'The two-buffer draft and active V4 are unchanged. Correct event ordering in this harness does not automatically modify or validate production call sites.',
            'No unsynchronized control is used, because race success/failure would not establish correctness. No performance measurement, router/GEMM execution, full model, startup or H200 validation.','']
        (self.out/'SUMMARY.txt').write_text('\n'.join(lines),encoding='utf-8')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.out.iterdir()) if p.is_file() and p.name!='checksums.json'}
        (self.out/'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        archive=self.out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.out.iterdir()):
                if p.is_file():z.write(p,p.name)
        print('\n'.join(lines));print('Download:',archive);return archive
