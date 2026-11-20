from pathlib import Path
P=Path(__file__).resolve().parent;p=P/'build_reuse_prototype.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("for f in ('virtual_k_context.cuh','virtual_k_epilogue.cuh'):(P/f).write_bytes((OLD/f).read_bytes())", """(P/'virtual_k_context.cuh').write_bytes((OLD/'virtual_k_context.cuh').read_bytes())
writer=(V/'include/humming/epilogue/smem_writer.cuh').read_text(encoding='utf-8-sig').replace('class EpilogueSmemWriter','class ResearchVirtualKSmemWriter').replace('  EpilogueSmemWriter(Ctx &ctx, ArithClass &arith)','  ResearchVirtualKSmemWriter(Ctx &ctx, ArithClass &arith)').replace('K_WARPS = BlockShape::K / WarpShape::K','K_WARPS = Ctx::K_WARPS')
(P/'virtual_k_smem_writer.cuh').write_text(writer,encoding='utf-8')
epi=(OLD/'virtual_k_epilogue.cuh').read_text(encoding='utf-8-sig').replace('#include <humming/epilogue/smem_writer.cuh>','#include "virtual_k_smem_writer.cuh"').replace('using SmemWriter = EpilogueSmemWriter<','using SmemWriter = ResearchVirtualKSmemWriter<')
(P/'virtual_k_epilogue.cuh').write_text(epi,encoding='utf-8')""")
s=s.replace("s=s.replace(\"'mma_type':'wgmma'\"", "s=s.replace(\"'virtual_k_epilogue.cuh')\",\"'virtual_k_epilogue.cuh','virtual_k_smem_writer.cuh')\")\ns=s.replace(\"'mma_type':'wgmma'\"")
p.write_text(s,encoding='utf-8')
