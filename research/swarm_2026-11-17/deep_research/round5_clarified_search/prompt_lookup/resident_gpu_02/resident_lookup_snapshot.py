"""GPU resident request-local history experiment; no serving integration.

Fixed8 proposals. Token history stores target-confirmed tokens including anchor.
Caller owns unique request slots/epochs and stream order. Overflow poisons slot;
it is never silently accepted. No heuristic based on load or other requests.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def scan_history(H,L,Slots,Enabled,Epochs,Expected,Partial,CAP:tl.constexpr,NREQ:tl.constexpr,TILES:tl.constexpr,
                 MIN:tl.constexpr,MAX:tl.constexpr,EOS0:tl.constexpr,EOS1:tl.constexpr,
                 BLOCK:tl.constexpr):
    row=tl.program_id(0);tile=tl.program_id(1)
    slot=tl.load(Slots+row);slot_ok=(slot>=0)&(slot<NREQ)
    n=tl.load(L+slot,mask=slot_ok,other=0)
    live=(tl.load(Enabled+slot,mask=slot_ok,other=0)!=0)&(slot_ok&(tl.load(Epochs+slot,mask=slot_ok,other=-1)==tl.load(Expected+row)))
    end=tile*BLOCK+tl.arange(0,BLOCK)
    valid=live & (n>=MIN) & (n<=CAP) & (end>=MIN) & (end+7<=n)
    same=valid
    for k in tl.static_range(1,MIN+1):
        a=tl.load(H+slot*CAP+end-k,mask=valid,other=-2)
        b=tl.load(H+slot*CAP+n-k,mask=live & (n>=MIN) & (n<=CAP),other=-3)
        same=same & (a==b)
    match=tl.where(same,MIN,0)
    for k in tl.static_range(MIN+1,MAX+1):
        a=tl.load(H+slot*CAP+end-k,mask=valid & (end>=k),other=-2)
        b=tl.load(H+slot*CAP+n-k,mask=live & (n>=k) & (n<=CAP),other=-3)
        same=same & (end>=k) & (n>=k) & (a==b)
        match=tl.where(same,k,match)
    no_stop=tl.full((BLOCK,),True,tl.int1)
    for j in tl.static_range(7):
        token=tl.load(H+slot*CAP+end+j,mask=valid,other=-2)
        no_stop=no_stop & (token!=EOS0) & (token!=EOS1)
    score=tl.where(valid & (match>=MIN) & no_stop,match.to(tl.int64)*(CAP+1)+end,-1)
    tl.store(Partial+row*TILES+tile,tl.max(score,0))

@triton.jit
def finish_lookup(H,L,Slots,Enabled,Epochs,Expected,Partial,Tokens,Hit,CAP:tl.constexpr,NREQ:tl.constexpr,TILES:tl.constexpr,TB:tl.constexpr):
    row=tl.program_id(0);slot=tl.load(Slots+row)
    slot_ok=(slot>=0)&(slot<NREQ)
    n=tl.load(L+slot,mask=slot_ok,other=0);enabled=(tl.load(Enabled+slot,mask=slot_ok,other=0)!=0)&(slot_ok&(tl.load(Epochs+slot,mask=slot_ok,other=-1)==tl.load(Expected+row)))
    t=tl.arange(0,TB)
    score=tl.max(tl.load(Partial+row*TILES+t,mask=t<TILES,other=-1),0)
    hit=(score>=0)&enabled&(n>0)&(n<=CAP)
    end=score%(CAP+1)
    col=tl.arange(0,8)
    source=tl.where(col==0,n-1,end+col-1)
    value=tl.load(H+slot*CAP+source,mask=((col==0)&enabled&(n>0)&(n<=CAP))|((col>0)&hit),other=0)
    tl.store(Tokens+row*8+col,value)
    tl.store(Hit+row,hit)

@triton.jit
def append_verified(H,L,Slots,Enabled,Epochs,Expected,Out,Commit,Status,CAP:tl.constexpr,NREQ:tl.constexpr):
    row=tl.program_id(0);slot=tl.load(Slots+row)
    slot_ok=(slot>=0)&(slot<NREQ)
    n=tl.load(L+slot,mask=slot_ok,other=0);count=tl.load(Commit+row)
    same_epoch=slot_ok&(tl.load(Epochs+slot,mask=slot_ok,other=-1)==tl.load(Expected+row))
    enabled=tl.load(Enabled+slot,mask=slot_ok,other=0)!=0
    ok=same_epoch&enabled&(n>0)&(count>=1)&(count<=8)&(n+count<=CAP)
    col=tl.arange(0,8)
    token=tl.load(Out+row*8+col,mask=col<count,other=0)
    tl.store(H+slot*CAP+n+col,token,mask=ok&(col<count))
    tl.store(L+slot,tl.where(ok,n+count,n),mask=same_epoch)
    tl.store(Enabled+slot,ok,mask=same_epoch)
    tl.store(Status+row,ok)

class ResidentLookup:
    def __init__(self,histories, *, capacity, eos_ids=(), device='cuda'):
        if not histories or any(not h or len(h)>capacity for h in histories):raise ValueError('invalid history/capacity')
        if capacity<8 or len(eos_ids)>2:raise ValueError('unsupported capacity/EOS configuration')
        if any(type(t)is not int or t<0 or t>=2**31 for h in histories for t in h):raise ValueError('invalid int32 token')
        self.capacity=capacity;self.rows=len(histories)
        self.history=torch.full((self.rows,capacity),-777,dtype=torch.int32,device=device)
        self.lengths=torch.tensor([len(h) for h in histories],dtype=torch.int32,device=device)
        for i,h in enumerate(histories):self.history[i,:len(h)].copy_(torch.tensor(h,dtype=torch.int32,device=device))
        self.enabled=torch.ones(self.rows,dtype=torch.int32,device=device)
        self.epochs=torch.zeros(self.rows,device=device,dtype=torch.int64)
        self.expected_epochs=torch.zeros(self.rows,device=device,dtype=torch.int64)
        self.slots=torch.arange(self.rows,device=device,dtype=torch.int32)
        self.tiles=triton.cdiv(capacity,256)
        self.partial=torch.empty((self.rows,self.tiles),device=device,dtype=torch.int64)
        self.tokens=torch.empty((self.rows,8),device=device,dtype=torch.int64)
        self.hit=torch.empty(self.rows,device=device,dtype=torch.int32)
        self.status=torch.empty(self.rows,device=device,dtype=torch.int32)
        self.eos=tuple(eos_ids)+(-1,)*(2-len(eos_ids))

    def bind_requests(self, slots, expected_epochs):
        # Called outside capture on the owning stream, before launching this batch.
        # Caller must finish prior users before changing shared captured bindings.
        if len(slots)!=self.rows or len(set(slots))!=self.rows:
            raise ValueError('one unique slot per request row required')
        if any(type(x)is not int or not 0<=x<self.rows for x in slots):
            raise ValueError('invalid request slot')
        if len(expected_epochs)!=self.rows or any(type(x)is not int or x<0 for x in expected_epochs):
            raise ValueError('invalid expected epoch')
        self.slots.copy_(torch.tensor(slots,device=self.history.device,dtype=torch.int32))
        self.expected_epochs.copy_(torch.tensor(expected_epochs,device=self.history.device,dtype=torch.int64))

    def lookup(self):
        scan_history[(self.rows,self.tiles)](self.history,self.lengths,self.slots,self.enabled,self.epochs,self.expected_epochs,self.partial,
             self.capacity,self.rows,self.tiles,4,16,*self.eos,256,num_warps=4)
        finish_lookup[(self.rows,)](self.history,self.lengths,self.slots,self.enabled,self.epochs,self.expected_epochs,self.partial,self.tokens,self.hit,
             self.capacity,self.rows,self.tiles,triton.next_power_of_2(self.tiles),num_warps=4)
        return self.tokens,self.hit

    def append(self,out,commit):
        if out.shape!=(self.rows,8) or commit.shape!=(self.rows,):raise ValueError('bad commit shape')
        if out.dtype!=torch.int64 or commit.dtype!=torch.int32:raise ValueError('bad commit dtype')
        if out.device!=self.history.device or commit.device!=self.history.device:raise ValueError('bad commit device')
        if not out.is_contiguous() or not commit.is_contiguous():raise ValueError('contiguous commit required')
        append_verified[(self.rows,)](self.history,self.lengths,self.slots,self.enabled,self.epochs,self.expected_epochs,out,commit,self.status,self.capacity,self.rows)
        return self.status
