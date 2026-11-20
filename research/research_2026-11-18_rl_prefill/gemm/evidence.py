"""Record registered CUDA cubin and symbol for every measured dispatch."""
import hashlib
import json
from pathlib import Path
from humming.kernel.humming import HummingKernel
from humming.layer import HummingMethod

def observe(staged_index, report):
    seen={}
    def record(role, stage, kernel_id):
        key=(role,stage,kernel_id)
        if key not in seen:
            k=HummingKernel._id2kernel[kernel_id]
            seen[key]={"role":role,"stage":stage,"id":kernel_id,"symbol":k.kernel_name,
                       "sha256":hashlib.sha256(Path(k.kernel_filename).read_bytes()).hexdigest(),
                       "config":str(k.block_shape)+"/"+str(k.warp_shape)}
        report["native_kernels"]=list(seen.values())

    baseline=HummingMethod.forward_layer
    def traced_baseline(layer,**kw):
        stage=kw["sublayer_name"]
        meta=layer.humming_metas[stage]
        handles=HummingKernel.prepare_kernels(meta.to_str(),kw["compute_config"],kw["tuning_config"]).tolist()
        record("baseline",stage,handles[2])
        return baseline(layer,**kw)
    HummingMethod.forward_layer=staticmethod(traced_baseline)

    changed=staged_index.staged_index_forward
    def traced_candidate(layer,**kw):
        before=staged_index.kernel_handle.cache_info()
        out=changed(layer,**kw)
        after=staged_index.kernel_handle.cache_info()
        if after.hits+after.misses>before.hits+before.misses:
            stage=kw["sublayer_name"];meta=layer.humming_metas[stage]
            handle=staged_index.kernel_handle(HummingKernel.current_context(),meta.to_str(),
                json.dumps(kw["compute_config"],sort_keys=True),json.dumps(kw["tuning_config"],sort_keys=True))
            record("candidate",stage,handle.tolist()[0])
        return out
    staged_index.staged_index_forward=traced_candidate


