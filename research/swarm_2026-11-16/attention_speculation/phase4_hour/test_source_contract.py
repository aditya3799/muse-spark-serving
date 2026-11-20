"""Execute pinned metadata/graph/KV selection contracts with CPU tensor stubs."""
import ast,copy,dataclasses,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
from dflash_prefix_research import resolve_verify_prefix,spans_overlap
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
class Tensor:
    def __init__(self,data,shape=None):
        if shape is None:
            shape=(len(data),len(data[0])) if data and isinstance(data[0],list) else (len(data),)
            if len(shape)==2:data=sum(data,[])
        self.data=list(data);self.shape=tuple(shape)
    def numel(self):return len(self.data)
    def view(self,*shape):
        if shape==(-1,):shape=(len(self.data),)
        if -1 in shape:shape=tuple(len(self.data)//shape[0] if x==-1 else x for x in shape)
        assert self.numel()==__import__('math').prod(shape)
        return Tensor(self.data,shape)
    reshape=view
    def contiguous(self):return Tensor(self.data,self.shape)
    def __getitem__(self,item):
        if isinstance(item,tuple):
            rows,cols=item;assert rows==slice(None)
            width=self.shape[1];data=[self.data[i*width:(i+1)*width][cols] for i in range(self.shape[0])]
            return Tensor(data)
        return Tensor(self.data[item])

def extract(path,name):
    source=path.read_text(encoding='utf-8')
    try:
        tree=ast.parse(source)
    except SyntaxError:
        # Native Python3.8 cannot parse unrelated parenthesized with-statements.
        # Parse this exact method's source only, without editing its contents.
        import re,textwrap
        lines=source.splitlines(True)
        start=next(i for i,l in enumerate(lines) if re.match(r'    def '+name+r'\(',l))
        end=next((i for i in range(start+1,len(lines)) if re.match(r'    (def |@)',lines[i])),len(lines))
        tree=ast.parse(textwrap.dedent(''.join(lines[start:end])))
    return next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def execute(nodes,namespace):
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations',asname=None)],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[]))
    exec(compile(module,'<pinned contract>','exec'),namespace)
files={
 'plan':ROOT/'sglang/python/sglang/srt/mem_cache/kv_loc_plan.py',
 'verify':ROOT/'sglang/python/sglang/srt/speculative/dflash_info.py',
 'graph':ROOT/'sglang/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py',
 'output':ROOT/'sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py'}
ns={'torch':NS(Tensor=Tensor),'TokenSpan':type('TokenSpan',(),{})}
execute([extract(files['plan'],'_cols')],ns)
class Base:
    def __init__(self,**kw):self.__dict__.update(kw)
verify_ns={'dataclass':dataclasses.dataclass,'SpecInput':Base,
           'CaptureHiddenMode':NS(FULL='full'),'SpecInputType':NS(DFLASH_VERIFY='verify'),'IdSpaceKind':NS(FULL='full')}
execute([extract(files['verify'],'DFlashVerifyInput')],verify_ns)
verify_cls=verify_ns['DFlashVerifyInput']
can_run=extract(files['graph'],'can_run_graph')
guard=next(n for n in can_run.body if isinstance(n,ast.If) and 'captured_req_width' in ast.dump(n.test) and 'num_tokens_per_req' in ast.dump(n.test))
graphfn=ast.parse('def graph_guard(self,spec_info):\n    return True\n').body[0]
graphfn.body=[guard,ast.Return(value=ast.Constant(value=True))]
graphns={};execute([graphfn],graphns)
outns={};execute([extract(files['output'],'_get_speculative_output_stride')],outns)
checks=0
for bs in (1,2,4,32):
    tokens=Tensor([[b*100+c for c in range(8)] for b in range(bs)])
    positions=Tensor([b*50+c for b in range(bs) for c in range(8)])
    targetids=Tensor([1000+b*128+c for b in range(bs) for c in range(8)])
    draftids=Tensor([90000+b*128+c for b in range(bs) for c in range(8)])
    for k in (1,2,4,6,8):
        width,candidates,pos,slots,cols=resolve_verify_prefix(tokens,positions,targetids,eligible=True,requested_width=k)
        expected=[b*100+c for b in range(bs) for c in range(k)]
        assert candidates.data==expected and width==k
        assert pos.data==[b*50+c for b in range(bs) for c in range(k)]
        dummyplan=NS(req_pool_indices=Tensor(list(range(bs))))
        assert ns['_cols'](dummyplan,targetids,cols).data==slots.data
        draft=ns['_cols'](dummyplan,draftids,cols)
        assert draft.data==[90000+b*128+c for b in range(bs) for c in range(k)]
        assert tokens.data==[b*100+c for b in range(bs) for c in range(8)]
        verify=verify_cls(draft_token=candidates.view(-1),positions=pos,draft_token_num=k,kv_loc_plan=dummyplan,kv_loc_cols=cols)
        assert verify.num_tokens_per_req==k and verify.num_tokens_for_logprob_per_req==k
        assert graphns['graph_guard'](NS(captured_req_width=8),verify)==(k==8)
        assert graphns['graph_guard'](NS(captured_req_width=k),verify)
        assert outns['_get_speculative_output_stride'](NS(speculative_output_stride=None,speculative_num_draft_tokens=8))==8
        checks+=1
    # Eligibility/default/unsupported widths retain original tensor identity.
    for kwargs in ({'eligible':False,'requested_width':4},{'eligible':True},{'eligible':True,'requested_width':3}):
        value=resolve_verify_prefix(tokens,positions,targetids,**kwargs)
        assert value[0]==8 and value[1] is tokens and value[-1] is None;checks+=1
# Execute the exact source-integrated eligibility expression and turn each guard off.
candidate=P/'candidate_sources/sglang/python/sglang/srt/speculative/dflash_worker_v2.py'
tree=ast.parse(candidate.read_text(encoding='utf-8'))
assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='prefix_eligible' for x in n.targets))
code=compile(ast.fix_missing_locations(ast.Expression(assignment.value)),'<eligibility>','eval')
for guardname in ('eligible','logits_gate','mamba','dp','simulation','missing_host_lens','route_output','embed_override'):
    worker=NS(_can_skip_verify_logits=lambda *x:guardname!='logits_gate',_need_mamba_verify_commit=guardname=='mamba')
    batch=NS(sampling_info=None,seq_lens_cpu=None if guardname=='missing_host_lens' else object(),reqs=[NS(return_routed_experts=guardname=='route_output')],replace_embeds=object() if guardname=='embed_override' else None)
    env={'self':worker,'batch':batch,'get_parallel':lambda:NS(attn_dp_enabled=guardname=='dp'),'SIMULATE_ACC_LEN':1 if guardname=='simulation' else 0}
    assert eval(code,env)==(guardname=='eligible');checks+=1
for args,expected in [((100,16,108,16),True),((108,16,100,16),True),((100,16,116,8),False),((100,0,100,16),False),((100,16,100,16),True),((100,32,108,4),True)]:
    assert spans_overlap(*args)==expected;checks+=1
report={'status':'PASS','checks':checks,'real_source_contracts':['KVLocPlan._cols','DFlashVerifyInput construction/__post_init__','DecodeCudaGraphRunner exact width guard','scheduler output stride','integrated eligibility expression'],
        'scope':'CPU tensor stubs execute pinned source contracts; not full scheduler, model, allocator or graph numerical execution.',
        'source_hashes':{k:hashlib.sha256(f.read_bytes()).hexdigest() for k,f in files.items()},
        'candidate_worker_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()}
(P/'cpu_source_contract_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
