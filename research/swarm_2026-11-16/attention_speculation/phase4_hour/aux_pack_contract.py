"""Isolated direct-cast shared prefill aux contract; not installed in serving.
Real runner integration must bind one owner before capture and keep a lease
through DFlash materialization. Unsupported configurations use original list.
"""
class AuxOutputOwner:
    def __init__(self,torch,max_tokens,hidden,capture_ids,device):
        self.torch=torch;self.max_tokens=max_tokens;self.hidden=hidden
        self.capture_ids=tuple(capture_ids)
        self.buffer=torch.empty((max_tokens,hidden*len(capture_ids)),dtype=torch.bfloat16,device=device)
        self.busy=False;self.generation=0
    def begin(self,tokens):
        if self.busy:raise RuntimeError('Aux consumer has not released prior target output')
        if not 0<tokens<=self.max_tokens:raise ValueError('Unsupported graph bucket')
        self.busy=True;self.generation+=1
        return AuxCapture(self,tokens,self.generation)
    def adopt_bcg_output(self,output,capacity_rows):
        # BCG hook: reuses a runner-owned persistent output, never graph-local
        # storage. Caller falls back to existing allocation when None is returned.
        if (not 0<capacity_rows<=self.max_tokens or output.ndim!=2
            or not 0<output.shape[0]<=capacity_rows
            or output.dtype!=self.buffer.dtype or output.device!=self.buffer.device
            or output.shape[1]!=self.buffer.shape[1]
            or output.data_ptr()!=self.buffer.data_ptr() or not output.is_contiguous()):
            return None
        return self.buffer[:capacity_rows]
    def release_after_materialization(self,generation):
        if not self.busy or generation!=self.generation:raise RuntimeError('Stale aux release')
        # On CUDA the integration must also guarantee consumer stream ordering;
        # host release alone is not evidence that asynchronous reads completed.
        self.busy=False

class AuxCapture:
    def __init__(self,owner,tokens,generation):
        self.owner=owner;self.tokens=tokens;self.generation=generation;self.index=0
    def capture(self,layer_id,stream):
        o=self.owner
        if not o.busy or self.generation!=o.generation:raise RuntimeError('Stale capture')
        if self.index>=len(o.capture_ids) or layer_id!=o.capture_ids[self.index]:raise ValueError('Capture order/count mismatch')
        if stream.shape!=(self.tokens,o.hidden) or stream.dtype!=o.torch.float32 or stream.device!=o.buffer.device:raise ValueError('Unsupported residual stream')
        # Copy performs the same FP32->BF16 rounding as .to(BF16), without the
        # temporary BF16 tensor and later cat. Slots are strided feature views.
        o.buffer[:self.tokens,self.index*o.hidden:(self.index+1)*o.hidden].copy_(stream)
        self.index+=1
    def finish(self):
        if not self.owner.busy or self.generation!=self.owner.generation:raise RuntimeError('Stale aux finish')
        if self.index!=len(self.owner.capture_ids):raise RuntimeError('Missing aux tap')
        return self.owner.buffer[:self.tokens]

def make_aux_owner(torch,*,max_tokens,hidden,capture_ids,num_layers,device,phase='prefill',sp=False,dp=1,pp=1):
    try:ids=tuple(capture_ids)
    except TypeError:return None
    if (any(not isinstance(v,int) or isinstance(v,bool) or v<1 for v in (max_tokens,hidden,num_layers))
        or any(not isinstance(i,int) or isinstance(i,bool) or not 0<=i<num_layers for i in ids)):
        return None
    if (phase!='prefill' or sp or dp!=1 or pp!=1 or not ids
        or ids!=tuple(sorted(set(ids)))):
        return None
    return AuxOutputOwner(torch,max_tokens,hidden,ids,device)
