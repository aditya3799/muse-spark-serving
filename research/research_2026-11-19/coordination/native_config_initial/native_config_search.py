"""Conservative native Humming pipeline search, independent of any serving workload.

Only stage count and launch residency vary. Operand formats, block/warp/K shapes,
FP32 accumulation, topology-aware light/heavy partition and all math stay native.
This research module does not activate itself in a model or serving process.
"""
import copy,hashlib,json,math,statistics
import torch


def config_variant(configs,stages,ctas):
    result=copy.deepcopy(configs)
    for intervals in result.values():
        for lo,hi,c in intervals:
            if c.get('use_f16_accum') is not False or c.get('use_stream_k') is not False:
                raise ValueError('requires existing FP32, non-StreamK policy')
            # TMA/warp-specialized variants need independent metadata qualification.
            if any(c.get(key,False) for key in ('use_tma_a','use_tma_b','use_tma_c','use_tma_as','use_tma_bs','use_warp_specialization')):
                raise ValueError('this first search family preserves cp.async only')
            c['num_stages']=stages
            c['num_ctas_per_sm']=ctas
    return result


def candidates(configs):
    seen={json.dumps(configs,sort_keys=True)}
    yield 'parent',copy.deepcopy(configs)
    for stages in (3,4):
        for ctas in (1,2):
            try:variant=config_variant(configs,stages,ctas)
            except ValueError:continue
            fingerprint=json.dumps(variant,sort_keys=True)
            if fingerprint in seen:continue
            seen.add(fingerprint)
            yield f'cpasync_stages{stages}_cta{ctas}',variant


def graph_arm(layer,configs,run):
    # copy.copy(Module) shares canonical parameters/buffers, with independent
    # policy in its top-level namespace. No new persistent weight allocation.
    view=copy.copy(layer)
    view.fp32_configs=copy.deepcopy(configs)
    run(view);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output=run(view)
    return {'view':view,'graph':graph,'output':output}


def graph_us(arm):
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record();arm['graph'].replay();end.record();end.synchronize()
    return begin.elapsed_time(end)*1000


def compare_arms(parent,candidate,update,seed,rounds=6):
    cells=[]
    for profile in ('uniform','clustered','all_local','all_remote'):
        ratios=[];pt=[];ct=[];exact=True
        for i in range(rounds):
            update(seed+7919*i,profile)
            order=(parent,candidate) if i%2==0 else (candidate,parent)
            times={id(arm):graph_us(arm) for arm in order}
            exact &= torch.equal(parent['output'],candidate['output'])
            pt.append(times[id(parent)]);ct.append(times[id(candidate)])
            ratios.append(pt[-1]/ct[-1])
        cells.append(dict(profile=profile,exact=exact,parent_us=pt,candidate_us=ct,
                          gain=math.exp(statistics.mean(map(math.log,ratios)))))
    return dict(exact=all(c['exact'] for c in cells),cells=cells,
                gain=math.exp(statistics.mean(math.log(c['gain']) for c in cells)),
                worst_gain=min(c['gain'] for c in cells))


def search(layer,run,update,seed=193781):
    original=copy.deepcopy(layer.fp32_configs)
    identity=hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()
    parent=graph_arm(layer,original,run)
    control=graph_arm(layer,original,run)
    trials=[];chosen=('parent',original,1.)
    for name,configuration in candidates(original):
        if name=='parent':continue
        row=dict(name=name,configuration=configuration)
        try:
            arm=graph_arm(layer,configuration,run)
            row['discovery']=compare_arms(parent,arm,update,seed)
            if row['discovery']['exact'] and row['discovery']['gain']>1.02 and row['discovery']['worst_gain']>.98:
                row['holdout']=compare_arms(control,arm,update,seed+15485863)
                score=row['holdout']['gain']
                if row['holdout']['exact'] and row['holdout']['worst_gain']>.98 and score>chosen[2]:
                    chosen=(name,copy.deepcopy(configuration),score)
            del arm
        except Exception as exc:
            row['error']=repr(exc)
            # This standalone research harness must stop on illegal memory
            # accesses; no later numerical result is valid after a poisoned GPU.
            if 'illegal' in str(exc).lower() or 'device-side assert' in str(exc).lower():raise
        trials.append(row)
    assert layer.fp32_configs==original,'shared parent policy mutated'
    return dict(parent_policy_sha256=identity,chosen=chosen[0],chosen_config=chosen[1],
                heldout_gain=chosen[2],trials=trials,
                scope='Local native whole-FFN policy comparison; no H200 or serving claim. Allremote control is charged and can reject launch-heavy variants.')
