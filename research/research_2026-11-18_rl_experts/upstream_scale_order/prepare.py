"""Generate the pinned, private Humming PR 107 backport and measurement harnesses."""
from pathlib import Path
import shutil
import hashlib
import sys

here = Path(__file__).resolve().parent
root = here.parents[1]
source = root / "research_2026-11-18_rl_prefill/gemm"
vendor = root / "swarm_2026-11-16/moe/humming_vendor/humming"
old = (source / "staged_index.py").read_text()
imports = old[old.index("import json"):old.index("OLD_BEGIN")]
helper = old[old.index("class StagedIndexKernel"):].replace("StagedIndexKernel","ScaleOrderKernel").replace("staged_index_forward","scale_order_forward").replace('kwargs["valid_shape_m"]>=4096','kwargs["valid_shape_m"]>=1')
main = (vendor / "include/humming/arith/mainloop_arith.cuh").read_text()
lo = main.index("        scalar_t *dq_bs_scalar_ptr",main.index("void may_process_bs_before_apply_on_b"))
hi = main.index("        const scalar_t2 scale_factor",lo)
remove = main[lo:hi]
old_index = 'bs_f16_ptr[i] = this->num2num2(bs_half_ptr[i]);'
new_index = ('if constexpr (ElementBS::kBits == 8 && !kUseFusedE8m0Scale) {\n'
             '            auto *raw_scale = reinterpret_cast<scalar_t *>(dq_bs);\n'
             '            const uint32_t lane = j < 4 ? (j / 2) * 4 + i * 2 + j % 2 : j * 2 + i;\n'
             '            bs_f16_ptr[i] = this->num2num2(raw_scale[lane]);\n'
             '          } else {\n'
             '            bs_f16_ptr[i] = this->num2num2(bs_half_ptr[i]);\n'
             '          }')
assert main.count(old_index)==1
binding = """@lru_cache(maxsize=1)
def adapted_source():
    headers = Path(humming.__file__).resolve().parent / "include/humming"
    kernel = (headers / "kernel/humming.cuh").read_text()
    main = (headers / "arith/mainloop_arith.cuh").read_text()
    if hashlib.sha256(main.encode()).hexdigest() != HEADER_SHA:
        return None
    assert main.count(OLD_SWAP) == main.count(OLD_INDEX) == 1
    main = main.replace(OLD_SWAP, "").replace(OLD_INDEX, NEW_INDEX)
    return kernel.replace("#include <humming/arith/mainloop_arith.cuh>", main)
"""
impl = ('# SPDX-License-Identifier: Apache-2.0\n'
        '# Adapted from vLLM Humming PR 107, lesj0610, by selecting probed scale lanes in the pinned reader.\n'
        '# https://github.com/vllm-project/humming/pull/107\n'
        + "import hashlib\n" + imports + 'HEADER_SHA = '+repr(hashlib.sha256(main.encode()).hexdigest())+'\n'
        + "OLD_SWAP = "+repr(remove)+"\nOLD_INDEX = "+repr(old_index)+"\nNEW_INDEX = "+repr(new_index)+"\n"
        +binding+'\n'+helper)
(here / "scale_order.py").write_bytes(impl.encode())
# Borrow the same registered-cubin tracing and scored helper adaptation used for the prior GPU run.
def rename(x):
    return x.replace("staged_index", "scale_order").replace("staged_route_ids","scale_register_order").replace('agent="prefill"', 'agent="experts"')
build = rename((source/"build.py").read_text().split("vendor=root/")[0])
(here/"build.py").write_bytes(build.encode())
harness = rename((source/"paired.py").read_text())
harness = harness.replace('from fixture_tuning_cache import', 'sys.path.insert(0,str(root/"research_2026-11-18_rl_prefill/gemm"))\nfrom fixture_tuning_cache import')
harness = harness.replace('from evidence import observe','from scale_evidence import observe')
harness = harness.replace('args=parser.parse_args()', 'parser.add_argument("--stage-check", action="store_true")\nargs=parser.parse_args()')
harness = harness.replace("used=set()", "used=set()\nstage_inputs={}")
harness = harness.replace('    before=scale_order.kernel_handle.cache_info()', '    if args.stage_check and recording[0]:\n        stage_inputs.setdefault(kwargs["sublayer_name"],[]).append(dict(kwargs))\n    before=scale_order.kernel_handle.cache_info()')
harness = harness.replace("for m in args.rows:", "recording=[False]\nfor m in args.rows:")
assert 'recording=[False]' in harness
harness = harness.replace('    graphs={};outputs={}', '''    if args.stage_check:
        from stage_pair import check
        stage_inputs.clear();recording[0]=True
        run(True);torch.cuda.synchronize();recording[0]=False
        report.setdefault("stage_trials",{})[str(m)]=check(m,layer,stage_inputs,args.seed,args.pairs,scale_order.scale_order_forward)
        stage_inputs.clear();save()
    graphs={};outputs={}''')
assert 'from stage_pair import check' in harness
harness = harness.replace('with cached_threshold_for_preparation(here):', '''# Stamp every finite signed E4M3 scale on each expert. Both timed paths use these buffers.
finite_codes = torch.tensor([code for code in range(256) if code not in (127,255)],device="cuda",dtype=torch.uint8)
for stage in ("w13","w2"):
    getattr(raw,stage+"_weight_scale").view(torch.uint8)[:,:254,0].copy_(finite_codes)
report["finite_e4m3_scale_bytes_planted_per_expert"] = int(finite_codes.numel())
with cached_threshold_for_preparation(here):''')
(here/"paired.py").write_bytes(harness.encode())
(here/"scale_evidence.py").write_bytes(rename((source/"evidence.py").read_text()).encode())
(here/"offline.py").write_bytes(rename((source/"offline.py").read_text()).encode())
license_path=root.parent / "inference_research/humming/LICENSE"
shutil.copyfile(license_path,here/"LICENSE.Humming")


