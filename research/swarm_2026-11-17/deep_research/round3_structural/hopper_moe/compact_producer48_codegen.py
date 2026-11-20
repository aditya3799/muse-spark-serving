"""Offline-only producer-register allocation control; arithmetic and barriers unchanged."""
from compact_codegen import render_compact as base_render
OLD='constexpr uint32_t kLoadThreadRegisters = TuningConfig::kNumMathThreads > 256 || (TuningConfig::kNumCtasPerSm == 1 && ElementA::kBits != 16) ? 40 : 24;'
NEW='constexpr uint32_t kLoadThreadRegisters = 48;'
ASSERT='static_assert(TuningConfig::kNumThreads == 384 && TuningConfig::kNumMathThreads == 256 && TuningConfig::kNumLoadThreads == 128 && TuningConfig::kNumCtasPerSm == 1 && TuningConfig::kNumStages >= 3 && ElementA::kBits == 16, "Producer48 qualified geometry");'
def render_producer48(source,c,enabled=True):
 if not enabled:raise ValueError('Explicit producer48 control only')
 if not c.get('use_warp_spec') or not c.get('use_tma') or c.get('num_stages',0)<3 or c.get('num_ctas_per_sm',1)!=1:raise ValueError('Unqualified producer48 policy')
 block=c['block_shape'];warp=c['warp_shape']
 if any(x%y for x,y in zip(block,warp)) or 32*(block[0]//warp[0])*(block[1]//warp[1])*(block[2]//warp[2])!=256:raise ValueError('Exactly256 math threads required')
 out,meta=base_render(source,c,True)
 if out.count(OLD)!=1:raise ValueError('Expected exactly one warp-specialized register allocation anchor')
 changed=out.replace(OLD,NEW+'\n  '+ASSERT)
 assert changed.replace(NEW+'\n  '+ASSERT,OLD)==out
 meta.update(producer_register_control={'old':24,'new':48,'consumer_registers':224,'register_budget_per_CTA':64512,'new_load_registers':6144,'new_math_registers':57344,'scope':'Resource-control only, identical math/scheduling/barriers; consumer cap232?224; H200 execution pending'})
 return changed,meta
