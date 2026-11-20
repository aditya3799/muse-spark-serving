"""Model-side adapter for SGLang tensors and GroupCoordinator.

Construct after normal weight loading. Workers explicitly register the same global shapes
on every rank before warmup/capture; absent or unsupported shapes use the original layer.
"""
import torch
import torch.distributed as dist

from runtime_projection import discover


class ProjectionRouter:
    def __init__(self, original, group, *, empirical=False):
        if not isinstance(original,torch.nn.Linear) or original.bias is not None:
            raise TypeError("only a native, bias-free Linear can use this adapter")
        self.original=original
        self.group=group
        self.plans={}
        self.empirical=empirical

    def register(self, examples, *, discover_fn=discover):
        examples=list(examples)
        if not examples: raise ValueError("rank registration requires examples")
        key=tuple(examples[0].shape)
        if key in self.plans: raise ValueError("shape already registered")
        plan,metadata=discover_fn([(v,self.original.weight) for v in examples],self.group.world_size)
        available=torch.tensor(int(plan is not None),dtype=torch.int32)
        dist.all_reduce(available,op=dist.ReduceOp.MIN,group=self.group.cpu_group)
        if not int(available):
            if plan: plan.close()
            plan=None
        self.plans[key]=plan
        return metadata

    def __call__(self, x):
        plan=self.plans.get(tuple(x.shape))
        if plan is None: return self.original(x)
        # SGLang ordinary TP grants all group members the same logical input geometry.
        # Local pointer/stride/stream rejection still issues the same collective.
        local=plan.local(x,self.original.weight,self.group.rank_in_group)
        native=None
        if local is None:
            native=self.original(x)
            width=native.shape[1]//self.group.world_size
            local=native.narrow(1,width*self.group.rank_in_group,width).contiguous()
        assembled=self.group.all_gather(local,dim=-1)
        if self.empirical: return assembled
        if native is None: native=self.original(x)
        actual,reference=assembled.view(torch.int16),native.view(torch.int16)
        return torch.where((actual==reference).all(),actual,reference).view(torch.bfloat16)

    def close(self):
        for plan in self.plans.values():
            if plan: plan.close()
        self.plans.clear()

