"""Research facade: canonical WGS storage, one graph-stable transient weight arena.

This module is not wired into serving. It deliberately admits one explicitly
bound CUDA stream and owner thread; production multi-stream integration is pending.
"""
import copy
import threading
import weakref


def validate_resident(resident, torch):
    experts = getattr(resident, 'experts', None)
    if type(experts) is not int or not 1 <= experts <= 32:
        raise ValueError('requires 1..32 local experts')
    device = resident.w13.device
    if device.type != 'cuda':
        raise ValueError('resident weights must be CUDA')
    for stage, n in (('w13', 8192), ('w2', 4096)):
        weights = getattr(resident, stage)
        scales = getattr(resident, stage + '_post_scale')
        if (weights.dtype != torch.uint8 or not weights.is_contiguous() or
            weights.device != device or weights.numel() != experts * n * 2304 or
            scales.dtype != torch.bfloat16 or scales.shape != (experts,) or
            not scales.is_contiguous() or scales.device != device):
            raise ValueError('resident stage metadata does not match WGS Muse geometry')
    return experts, device


class SharedWeightWorkspace:
    """The object owns bytes; facades contain aliases and metadata only."""
    def __init__(self, experts, device, *, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        if type(experts) is not int or not 1 <= experts <= 32:
            raise ValueError('unsupported expert count')
        self.experts = experts
        self.device = self.torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('requires CUDA scratch')
        self.owner_thread = threading.get_ident()
        self.stream = self.torch.cuda.current_stream(self.device)
        self.packed = [self.torch.empty(experts*n*2304,device=self.device,
                                       dtype=self.torch.uint8) for n in (8192,4096)]
        self.locks = self.torch.zeros(max(2048,self.torch.cuda.get_device_properties(self.device).multi_processor_count*8),
                                     device=self.device,dtype=self.torch.int32)
        self.facades = weakref.WeakKeyDictionary()
        self.layouts = {}
        self.sealed = False
        self.closed = False
        self.bytes = sum(t.numel()*t.element_size() for t in (*self.packed,self.locks))
        if self.bytes > 906_100_000:
            raise RuntimeError('shared weight workspace exceeds explicit storage bound')

    def _check_context(self):
        if self.closed or threading.get_ident()!=self.owner_thread:
            raise RuntimeError('workspace lifetime/thread mismatch')
        if self.torch.cuda.current_stream(self.device)!=self.stream:
            raise RuntimeError('workspace stream mismatch')

    def _views(self, index, n):
        flat = self.packed[index]
        size = self.experts*n*2048
        weights = flat[:size].view(self.torch.int32).view(self.experts,256,n*2)
        scales = flat[size:].view(self.torch.float8_e4m3fn).view(self.experts,256,n)
        return weights,scales

    def prepare(self, resident):
        """Construct tiny metadata outside capture; never transform or copy weights."""
        self._check_context()
        if resident in self.facades:
            return self.facades[resident]
        if self.sealed or self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('new layer metadata must be prepared before capture')
        experts,device = validate_resident(resident,self.torch)
        if experts!=self.experts or device!=self.device:
            raise ValueError('resident layer mismatches shared workspace')
        from humming import dtypes
        from humming.config import WeightScale2Type,WeightScaleType
        from humming.layer import HummingMethod
        from humming.schema import HummingInputSchema,HummingWeightSchema
        facade = self.torch.nn.Module()
        layouts = {}
        for j,(stage,n) in enumerate((('w13',8192),('w2',4096))):
            weights,scales=self._views(j,n)
            facade.register_parameter(stage+'_weight',self.torch.nn.Parameter(weights,requires_grad=False))
            facade.register_parameter(stage+'_weight_scale',self.torch.nn.Parameter(scales,requires_grad=False))
            facade.register_buffer(stage+'_post_scale',getattr(resident,stage+'_post_scale'))
            schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,
                weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,
                weight_scale_2_type=WeightScale2Type.NONE)
            meta=HummingMethod.prepare_layer_meta(facade,shape_n=n,shape_k=4096,
                weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),
                num_experts=experts,torch_dtype=self.torch.bfloat16,sublayer_name=stage)
            if (meta.a_dtype!=dtypes.bfloat16 or meta.c_dtype!=dtypes.bfloat16 or
                meta.use_int_weight_scale or meta.has_input_scale or meta.use_fused_e8m0_scale or
                meta.should_apply_bs_on_c or meta.has_zero_point or meta.has_bias):
                raise ValueError('unsupported native metadata transform')
            layouts[stage]='humming_wgmma' if 'wgmma' in str(meta.mma_type).lower() else 'humming_mma'
        facade.register_buffer('locks',self.locks)
        facade.fp32_configs={}
        for stage in ('w13','w2'):
            configs=HummingMethod.get_default_tuning_configs(facade,use_f16_accum=False,
                                                            gemm_type='indexed',sublayer_name=stage)
            facade.fp32_configs[stage]=copy.deepcopy(configs)
            for _,_,config in facade.fp32_configs[stage]:
                config['use_stream_k']=False
                config['use_f16_accum']=False
        from sglang.kernels.ops.moe.muse_spark_humming_split import prepare_large_prefill_partition
        prepare_large_prefill_partition(facade,self.torch)
        self.facades[resident]=facade
        self.layouts[facade]=layouts
        return facade

    def seal_for_capture(self):
        self._check_context()
        self.sealed=True

    def convert(self,resident,layout='native'):
        self._check_context()
        if resident not in self.facades:
            raise RuntimeError('resident not prepared')
        from wgs_bridge import convert
        facade=self.facades[resident]
        operands={}
        for j,(stage,n) in enumerate((('w13',8192),('w2',4096))):
            target=self.layouts[facade][stage] if layout=='native' else layout
            operands[stage]=convert(getattr(resident,stage),self.packed[j],n,
                                   gate_up=stage=='w13',layout=target)
        return facade,operands

    def project(self,resident,helper,x,ids,*,defer_downscale=False,global_experts=256):
        facade,_=self.convert(resident,'native')
        return helper(facade,x,ids,self.torch,retain=False,
                      defer_downscale=defer_downscale,global_experts=global_experts)

    def close(self):
        self._check_context()
        if self.sealed or self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('graph-owned shared workspace cannot be closed')
        self.torch.cuda.synchronize(self.device)
        self.facades.clear();self.layouts.clear()
        self.packed=[];self.locks=None;self.closed=True
