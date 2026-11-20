"""Check cross-stream external graph events using CUDA Driver API, avoiding Torch startup."""
import ctypes as c
import json
import os
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent/"decode_overlap"
u64=c.c_uint64;u32=c.c_uint32;ptr=c.c_void_p
def main():
    start=time.monotonic()
    if os.name=="nt":dll=c.WinDLL("nvcuda.dll")
    else:dll=c.CDLL("libcuda.so.1")
    def run(symbol,args,*values):
        fn=getattr(dll,symbol)
        fn.restype=c.c_int;fn.argtypes=args
        code=fn(*values)
        if code:raise RuntimeError(symbol,code)
    run("cuInit",[u32],0)
    device=c.c_int();run("cuDeviceGet",[c.POINTER(c.c_int),c.c_int],c.byref(device),0)
    ctx=ptr();run("cuDevicePrimaryCtxRetain",[c.POINTER(ptr),c.c_int],c.byref(ctx),device)
    run("cuCtxSetCurrent",[ptr],ctx)
    def new_stream():
        s=ptr();run("cuStreamCreate",[c.POINTER(ptr),u32],c.byref(s),1);return s
    def new_event():
        e=ptr();run("cuEventCreate",[c.POINTER(ptr),u32],c.byref(e),2);return e
    def alloc(n):
        x=u64();run("cuMemAlloc_v2",[c.POINTER(u64),c.c_size_t],c.byref(x),n*4);return x
    def fill(x,v,n,s):
        run("cuMemsetD32Async",[u64,u32,c.c_size_t,ptr],x,v,n,s)
    def copy(x,y,n,s):
        run("cuMemcpyDtoDAsync_v2",[u64,u64,c.c_size_t,ptr],x,y,n*4,s)
    def wait(s,e):
        run("cuStreamWaitEvent",[ptr,ptr,u32],s,e,0)
    forward,projection=new_stream(),new_stream()
    temp=alloc(1048576)
    graphs=[]
    for n in (64,128):
        src,auxout,post,out=(alloc(n) for _ in range(4))
        ready,done=new_event(),new_event();graph=ptr();ex=ptr()
        run("cuStreamBeginCapture_v2",[ptr,c.c_int],forward,0)
        # Pre-marker memory traffic gives the consumer an opportunity to observe
        # stale state if it was accidentally waiting for the previous graph launch.
        fill(temp,1,1048576,forward);copy(auxout,src,n,forward)
        run("cuEventRecordWithFlags",[ptr,ptr,u32],ready,forward,1)
        fill(temp,2,1048576,forward);copy(post,src,n,forward)
        run("cuStreamEndCapture",[ptr,c.POINTER(ptr)],forward,c.byref(graph))
        run("cuGraphInstantiateWithFlags",[c.POINTER(ptr),ptr,u64],c.byref(ex),graph,0)
        graphs.append((n,src,auxout,post,out,ready,done,graph,ex))
    passed=0
    for i in range(24):
        n,src,auxout,post,out,ready,done,graph,ex=graphs[i%2]
        value=17+13*i
        fill(src,value,n,forward);fill(auxout,0xbadc0de,n,forward)
        run("cuGraphLaunch",[ptr,ptr],ex,forward)
        wait(projection,ready);copy(out,auxout,n,projection)
        run("cuEventRecord",[ptr,ptr],done,projection)
        wait(forward,done);fill(auxout,0xdeadbeef,n,forward)
        run("cuStreamSynchronize",[ptr],forward)
        host=(u32*n)()
        for actual in (out,post):
            run("cuMemcpyDtoH_v2",[ptr,u64,c.c_size_t],host,actual,4*n)
            assert all(x==value for x in host),(i,list(host[:2]))
            passed+=n
    result=dict(scope="Native CUDA driver event-node smoke, two real capture variants with a second stream and changed data. Four MiB pre/post marker memory touches are synchronization stress. No PyTorch project, model work or timing estimate.",
                executions=24,equal_uint32_values=passed,elapsed_seconds=time.monotonic()-start)
    (HERE/"driver_event_cuda.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)
    # This short process owns a primary-context reference; release it last.
    run("cuDevicePrimaryCtxRelease_v2",[c.c_int],device)
if __name__=="__main__":main()
