"""Compile only the shared-original-weight device kernel, using the judge's exact shipped headers."""
from pathlib import Path
import hashlib,json,subprocess,tempfile,time,resource,sys
import compile_fp8 as cfg
here=Path(__file__).resolve().parent;inc=here/"cutlass_judge/include"
lowopt="--lowopt" in sys.argv
label="normal_device_judge_t64"
rel=Path('cutlass/detail/collective/mixed_input_utils.hpp')
raw=(inc/rel).read_text()
assert hashlib.sha256(raw.encode()).hexdigest()=='6bc2a648b879b8555c4c3337021ac39906befa4aaa0865c47866058f698ae562'
needle='''      if constexpr (is_same_v<DstType, ElementScale>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));'''
assert raw.count(needle)==2
custom='''      if constexpr (is_same_v<ElementScale, uint32_t> && is_same_v<DstType, cutlass::bfloat16_t>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            const uint32_t word = scales_vm(j, i);
            const uint8_t byte = uint8_t(word >> (8 * k_block));
            const auto scale = cutlass::float_e4m3_t::bitcast(byte);
            const auto normalized_scale = cutlass::bfloat16_t(float(scale) * 0.25f);
            dst_vm(j, i) *= normalized_scale;
          }
        }
      }
      else '''+needle.lstrip()
with tempfile.TemporaryDirectory() as d:
 d=Path(d);p=d/rel;p.parent.mkdir(parents=True);p.write_text(raw.replace(needle,custom,1))
 out=here/(label+'_sm90a.cubin')
 args=[str(cfg.nvcc),'-std=c++17','-arch=sm_90a','--cubin','--expt-relaxed-constexpr','--expt-extended-lambda',('-O0' if lowopt else '-O2'),'-Xptxas=-O3,-v','-DPACKED_SCALE','-DPACKED_FP4','-DPROBE_TOKEN_TILE=64','-DPROBE_CTA_K=64',f'-I{d}',f'-I{inc}',*(f'-I{x}' for x in cfg.incs),str(here/'fp8_probe.cu'),'-o',str(out)]
 t=time.monotonic();r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (here/(label+'.log')).write_text(r.stdout)
 report=dict(optimization='O0 frontend/O3 ptxas' if lowopt else 'O2 frontend/O3 ptxas',returncode=r.returncode,seconds=time.monotonic()-t,peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,headers_commit='f3fde58372d33e9a5650ba7b80fc48b3b49d40c8',bytes=out.stat().st_size if out.exists() else 0,sha256=hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else None)
 (here/(label+'.json')).write_text(json.dumps(report,indent=2));print(report)
 print('\n'.join(s for s in r.stdout.splitlines() if 'error' in s.lower())[:1400])
 raise SystemExit(r.returncode)
