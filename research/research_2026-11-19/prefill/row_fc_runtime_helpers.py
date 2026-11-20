"""Additional deployment helpers appended to the standalone row adapter package."""
import fcntl
import contextlib
import hashlib
import inspect
import logging
import os
import statistics
import subprocess
import sysconfig
import tempfile

_row_logger=logging.getLogger(__name__)

def _compiled_library():
    source=HERE/'dflash_row_fc.cc'
    digest=hashlib.sha256(source.read_bytes()+str(torch.__version__).encode()+str(torch.version.cuda).encode()).hexdigest()[:20]
    directory=Path(tempfile.gettempdir())/'sglang-dflash-row-fc'/digest
    directory.mkdir(parents=True,exist_ok=True)
    output=directory/'row_plans.so'
    with (directory/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if output.is_file():return output
        nvidia=Path(sysconfig.get_paths()['purelib'])/'nvidia'
        location=next((p for p in (nvidia/'cu13',nvidia/'cublas') if (p/'include/cublasLt.h').is_file()),None)
        if location is None:raise RuntimeError('installed cuBLASLt headers unavailable')
        libraries=list((location/'lib').glob('libcublasLt.so.*'))
        if not libraries:raise RuntimeError('installed cuBLASLt library unavailable')
        command=[os.environ.get('CXX','c++'),'-O2','-std=c++17','-shared','-fPIC']
        for path in (location,nvidia/'cuda_runtime'):
            if (path/'include').is_dir():command+=['-I',str(path/'include')]
        temporary=directory/('row_plans.'+str(os.getpid())+'.so')
        command += [str(source),str(libraries[0]),'-Wl,-rpath,'+str(location/'lib'),'-o',str(temporary)]
        completed=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=90)
        if completed.returncode:raise RuntimeError(completed.stderr[-2000:])
        temporary.replace(output)
    return output

def _row_gather_output(group,m,n,weight):
    # Match GroupCoordinator.all_gather's allocator mode as well as its backend.
    # Some registered communicators require a symmetric output allocation.
    context=(group.use_symmetric_memory(group,disabled=not group.is_allocation_symmetric())
             if hasattr(group,'use_symmetric_memory') else contextlib.nullcontext())
    with context:return torch.empty((m,n),device=weight.device,dtype=weight.dtype)


def _row_startup_timing(router,m,k,generator,rounds=4):
    """Native FC versus full-K local FC and the actual TP all-gather.

    Output allocation and recoverable allocation votes precede timing. Native
    and gathered outputs never coexist, keeping the calibration memory bound.
    """
    group=router.group;weight=router.original.weight
    x=None;local=None;error=None
    try:
        x=torch.empty((m,k),device=weight.device,dtype=weight.dtype).normal_(generator=generator)
        local=torch.empty((m//group.world_size,weight.shape[0]),device=weight.device,dtype=weight.dtype)
    except Exception as exc:error=dict(message=repr(exc),fatal=not isinstance(exc,torch.OutOfMemoryError))
    declarations=[None]*group.world_size
    dist.all_gather_object(declarations,error,group=group.cpu_group)
    if any(declarations):
        if any(d is not None and d['fatal'] for d in declarations):raise RuntimeError('row timing initialization failure: '+repr(declarations))
        return dict(active=False,errors=declarations)
    plan=router.plans[(tuple(x.shape),_stream(x))]
    records={name:[] for name in ('native_us','local_us','gather_us','combined_us')}
    dispatch_ok=True
    for step in range(-1,rounds):
        for arm in (('native','candidate') if step%2==0 else ('candidate','native')):
            output=None;error=None
            events=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
            if arm=='native':
                try:
                    events[0].record();output=router.original(x);events[2].record();events[2].synchronize()
                except Exception as exc:error=dict(message=repr(exc),fatal=not isinstance(exc,torch.OutOfMemoryError))
            else:
                try:output=_row_gather_output(group,m,weight.shape[0],weight)
                except Exception as exc:error=dict(message=repr(exc),fatal=not isinstance(exc,torch.OutOfMemoryError))
            declarations=[None]*group.world_size
            dist.all_gather_object(declarations,error,group=group.cpu_group)
            if any(declarations):
                if any(d is not None and d['fatal'] for d in declarations):raise RuntimeError('row timing allocation/compute failure: '+repr(declarations))
                return dict(active=False,errors=declarations)
            if arm=='candidate':
                events[0].record()
                status=plan.invoke(plan.index,False,x.narrow(0,group.rank_in_group*local.shape[0],local.shape[0]),weight,local)
                events[1].record()
                # A rejected vendor dispatch still joins this diagnostic gather.
                # Its output is discarded and every rank disables the shape.
                group.all_gather_into_tensor(output,local)
                events[2].record();events[2].synchronize();dispatch_ok &= status==0
            if step>=0:
                if arm=='native':records['native_us'].append(events[0].elapsed_time(events[2])*1000)
                else:
                    records['local_us'].append(events[0].elapsed_time(events[1])*1000)
                    records['gather_us'].append(events[1].elapsed_time(events[2])*1000)
                    records['combined_us'].append(events[0].elapsed_time(events[2])*1000)
            del output
    return dict(active=bool(dispatch_ok),scope='Real loaded weights, synthetic calibration input, actual TP transport; not serving throughput',
                samples=records,medians={name:statistics.median(values) for name,values in records.items()})


def create_prefill_projection(model,group,capture_rows,*,ordinary_tp,measure=True):
    """Collective startup qualification; unsupported or failed shapes stay native.

    The environment switch is an explicit opt-in diagnostic. It never changes
    model precision, draft KV ownership, normalization or any cache policy.
    """
    enabled=os.environ.get('SGLANG_EXPERIMENTAL_DFLASH_ROW_FC','0')=='1'
    original=getattr(model,'fc',None)
    rows=[]
    local=dict(enabled=enabled,rows=rows,ordinary_tp=bool(ordinary_tp),measure=bool(measure),error=None)
    try:
        rows=sorted(set(int(m) for m in capture_rows if int(m)>=512 and int(m)%group.world_size==0));local['rows']=rows
        if not ordinary_tp or group.world_size not in (2,4,8):raise ValueError('requires ordinary replicated-input TP')
        if not isinstance(original,torch.nn.Linear) or original.bias is not None:raise ValueError('requires unchanged bias-free ordinary Linear')
        if not original.weight.is_cuda or original.weight.dtype!=torch.bfloat16:raise ValueError('requires CUDA BF16 weights')
        if 'projection' not in inspect.signature(model.project_target_hidden).parameters:raise ValueError('model projection contract unsupported')
        local['weight_shape']=tuple(original.weight.shape)
    except Exception as error:local['error']=repr(error)
    declarations=[None]*group.world_size
    dist.all_gather_object(declarations,local,group=group.cpu_group)
    if not all(d['enabled'] and d['error'] is None and d['measure']==declarations[0]['measure'] and d['rows']==declarations[0]['rows'] and d.get('weight_shape')==declarations[0].get('weight_shape') for d in declarations):return None
    # Compile/import failure is also voted before anyone starts GPU discovery.
    error=None
    try:_library();workspace=_workspace();assert workspace is not None
    except Exception as exc:error=repr(exc);workspace=None
    available=torch.tensor(int(error is None),dtype=torch.int32)
    dist.all_reduce(available,op=dist.ReduceOp.MIN,group=group.cpu_group)
    if not int(available):
        _row_logger.warning('DFlash row FC unavailable; native fallback: %s',error)
        return None
    n,k=original.weight.shape
    workspace_bytes=workspace[1]
    # One input + native reference + local output. Charge three workspaces
    # (native handle, discovered plan, prior retained plan) and32MiB margin.
    # Transport checks release the input before allocating gathered output.
    rows=[m for m in rows if m*(k*2+n*2+n*2//group.world_size)+3*workspace_bytes+(32<<20) < 1_800_000_000]
    router=RowProjectionRouter(original,group,empirical=True)
    accepted,_=router.prepare_registration([(m,k) for m in rows])
    if not accepted:return None
    gen=torch.Generator(device=original.weight.device).manual_seed(19261119)
    torch.cuda.synchronize(original.weight.device)
    allocated_before=torch.cuda.memory_allocated(original.weight.device)
    reserved_before=torch.cuda.memory_reserved(original.weight.device)
    peak_allocated_before=torch.cuda.max_memory_allocated(original.weight.device)
    peak_reserved_before=torch.cuda.max_memory_reserved(original.weight.device)
    try:
        for m in rows:
            x=None;failure=None
            try:x=torch.empty((m,k),device=original.weight.device,dtype=torch.bfloat16).normal_(generator=gen)
            except Exception as exc:failure=repr(exc)
            metadata=router.register([] if x is None else [x])
            del x;x=None
            if metadata.get('accepted'):
                # Validate real TP transport and each rank's actual replicated
                # checkpoint matrix on two further changing inputs.
                correct=True
                for mutation in ('magnitude','integer'):
                    x=None;reference=None;local_output=None;output=None;local_error=None
                    try:
                        x=torch.empty((m,k),device=original.weight.device,dtype=torch.bfloat16)
                        if mutation=='magnitude':x.normal_(generator=gen);x[:,::5].mul_(8)
                        else:x.random_(-2,3,generator=gen)
                        key=((m,k),_stream(x));plan=router.plans[key]
                        reference=original(x)
                        local_output=plan.local(x,original.weight,group.rank_in_group)
                        if local_output is None:
                            count=m//group.world_size
                            local_output=reference.narrow(0,group.rank_in_group*count,count).contiguous()
                        torch.cuda.synchronize(original.weight.device)
                        del x;x=None
                        # Allocate the entire collective output before voting;
                        # one-rank OOM cannot strand peers inside a GPU gather.
                        output=_row_gather_output(group,m,n,original.weight)
                    except Exception as exc:
                        local_error=dict(message=repr(exc),fatal=not isinstance(exc,(torch.OutOfMemoryError,ValueError,TypeError)))
                        del x;x=None
                    ready=torch.tensor(int(local_error is None),dtype=torch.int32)
                    dist.all_reduce(ready,op=dist.ReduceOp.MIN,group=group.cpu_group)
                    if not int(ready):
                        errors=[None]*group.world_size
                        dist.all_gather_object(errors,local_error,group=group.cpu_group)
                        del output,reference,local_output
                        if any(e is not None and e['fatal'] for e in errors):raise RuntimeError('collective row-FC startup CUDA failure: '+repr(errors))
                        correct=False;metadata['transport_allocation_errors']=errors
                        break
                    group.all_gather_into_tensor(output,local_output)
                    correct &= torch.equal(output.view(torch.int16),reference.view(torch.int16))
                    del output,reference,local_output
                valid=torch.tensor(int(correct),dtype=torch.int32)
                dist.all_reduce(valid,op=dist.ReduceOp.MIN,group=group.cpu_group)
                if not int(valid):
                    plan=router.plans.pop(key,None)
                    if plan:plan.close()
                    router.published.discard((m,k));metadata['accepted']=False
            if metadata.get('accepted') and measure:
                timing=_row_startup_timing(router,m,k,gen)
                timings=[None]*group.world_size
                dist.all_gather_object(timings,timing,group=group.cpu_group)
                faster=all(t.get('active') and t['medians']['combined_us']<t['medians']['native_us'] for t in timings)
                metadata['startup_timings']=timings
                if not faster:
                    plan=router.plans.pop(((m,k),_stream(original.weight)),None)
                    if plan:plan.close()
                    router.published.discard((m,k));metadata['accepted']=False
                    metadata['disable_reason']='actual all-rank local FC plus gather latency gate'
            if group.rank_in_group==0:_row_logger.info('DFlash row FC qualification M=%d accepted=%s metadata=%s allocation_error=%s',m,metadata.get('accepted'),metadata,failure)
            del x
        torch.cuda.synchronize(original.weight.device)
        _row_logger.info('DFlash row FC startup allocator history (no peak reset): %s',dict(
            allocated_before=allocated_before,reserved_before=reserved_before,
            peak_allocated_before=peak_allocated_before,peak_reserved_before=peak_reserved_before,
            allocated_after=torch.cuda.memory_allocated(original.weight.device),reserved_after=torch.cuda.memory_reserved(original.weight.device),
            peak_allocated_after=torch.cuda.max_memory_allocated(original.weight.device),peak_reserved_after=torch.cuda.max_memory_reserved(original.weight.device),
            extra_allocated_upper_bound=torch.cuda.max_memory_allocated(original.weight.device)-allocated_before,
            extra_reserved_upper_bound=torch.cuda.max_memory_reserved(original.weight.device)-reserved_before,
            upper_bound_includes_prior_global_peak=True,fixed_scratch_bytes=sum(t.numel() for t in router.workspaces.values())))
        torch.cuda.empty_cache()
        return router if router.published else None
    except Exception:
        # A fatal CUDA/collective error is not recoverable by silently changing
        # the collective sequence. Surface it as startup failure, never serve
        # with partially published per-rank state.
        raise
