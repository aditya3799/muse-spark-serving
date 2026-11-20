"""Render an isolated SGLang adapter for the compiled window entry; no submission."""
import ast
from pathlib import Path
from build_windows import D, PACKAGE, once

def main():
    source=(PACKAGE/"hopper_backend.py").read_text()
    source=once(source,"with_name(\"hopper_expert.cu\")","with_name(\"hopper_window.cu\")")
    source=once(source,'name="sglang_muse_hopper_expert"','name="sglang_muse_hopper_window_expert"')
    source=once(source,'    return lib\n', '''    lib.expert_run_window.argtypes = [ct.c_void_p]*10 + [ct.c_int]*3 + [ct.c_void_p]*3 + [ct.c_int,ct.c_void_p]
    return lib
''')
    first=source.index("            for first in range(0,rows,384):")
    stop=source.index("            if not defer_downscale:",first)
    existing=source[first:stop]
    window='''            if rows > 384:
                from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size

                slots, groups, counts = moe_align_block_size(ids, 64, self.experts, return_inactive_prefix_count=True)
                # Bounded windows share activation scratch. Empty windows return on device.
                args = (self.w13, self.w2, self.gate_scale, self.table, x, ids, out, counter, ready, tiles)
                pointers = tuple(tensor.data_ptr() for tensor in args)
                metadata = (slots.data_ptr(), groups.data_ptr(), counts.data_ptr())
                for window in range((groups.numel()+47)//48):
                    status = self.lib.expert_run_window(*pointers, rows, self.experts, self.grid, *metadata, window, stream)
                    if status:
                        raise RuntimeError(f"Hopper expert window launch failed with CUDA error {status}")
            else:
'''
    source=source[:first]+window+"".join("    "+line for line in existing.splitlines(keepends=True))+source[stop:]
    ast.parse(source)
    (D/"hopper_window_backend.py").write_text(source)
    (D/"hopper_pack.py").write_bytes((PACKAGE/"hopper_pack.py").read_bytes())
if __name__=="__main__":
    main()
