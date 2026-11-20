from pathlib import Path
H=Path(__file__).resolve().parent
p=H/"fixed_tree_adapter.py";s=p.read_text()
a=s.index("        self.tokens = alloc(shape");e=s.index("        self.ancestors",a)
s=s[:a]+'''        self.bind_topology(alloc((batch*WIDTH*5,),torch.int32))
'''+s[e:]
a=s.index("    def active(")
s=s[:a]+'''    def bind_topology(self, wire):
        self.wire=wire
        n=self.batch*WIDTH
        self.tokens=wire[:2*n].view(torch.int64).view(self.batch,WIDTH)
        self.parents=wire[2*n:3*n].view(self.batch,WIDTH)
        self.depth=wire[3*n:4*n].view(self.batch,WIDTH)
        self.mask=wire[4*n:5*n].view(torch.uint32).view(self.batch,WIDTH)

'''+s[a:]
s=s.replace('        for key,value in vars(self).items():','''        view.bind_topology(self.wire[:batch*WIDTH*5])
        for key,value in vars(self).items():
            if key in ("wire","tokens","parents","depth","mask"):
                continue''')
p.write_text(s)
p=H/"tree_dispatch.py";s=p.read_text().replace("    step.stage(table,batch.req_pool_indices",'''    # Keep the topology and the request-table permutation identical across ranks.
    # This is a single collective with one contiguous 160-byte payload per request.
    tp_group=get_parallel().tp_group
    if tp_group.world_size>1:
        tp_group.broadcast(step.wire,src=0)
    step.stage(table,batch.req_pool_indices''')
p.write_text(s)
p=H/"test_tree_dispatch.py";s=p.read_text().replace('par=N(attn_dp_enabled','par=N(tp_group=N(world_size=1),attn_dp_enabled')
s=s.replace('base_ptr=w._fixed_tree_slab.full.data_ptr()','''wire=step.wire.clone()
calls=[]
def restore(data,src):
    assert src==0
    data.copy_(wire)
    calls.append(data.numel()*data.element_size())
par.tp_group=N(world_size=8,broadcast=restore)
# A locally incorrect rank proposal must become the shared rank-zero proposal before staging.
saved=w._draft_sampler.candidate_out
w._draft_sampler.candidate_out=saved+123
repaired=active()
assert torch.equal(repaired.wire,wire)
assert torch.equal(repaired.tokens,wire[:2*3*8].view(torch.int64).view(3,8))
w._draft_sampler.candidate_out=saved
assert calls==[480]
par.tp_group=N(world_size=1)
base_ptr=w._fixed_tree_slab.full.data_ptr()''').replace('if torch.is_tensor(v)}','if torch.is_tensor(v) and k not in ("tokens","parents","depth","mask")}')
s=s.replace("r=dict(guarded_fallbacks","r=dict(tp_rank_repaired_wire_bytes=calls[0],guarded_fallbacks")
p.write_text(s)
p=H/"fixed_tree_gpu.py";s=p.read_text().replace('sum(v.numel()*v.element_size() for v in vars(step).values() if torch.is_tensor(v))','sum(v.numel()*v.element_size() for k,v in vars(step).items() if torch.is_tensor(v) and k not in ("tokens","parents","depth","mask"))')
p.write_text(s)

