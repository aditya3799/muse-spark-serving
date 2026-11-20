"""Independent CPU proof against literal Humming pack loops and donor tensor axes."""
import ast, hashlib, json, time
from pathlib import Path
from canonical_tile_adapter import humming_code_address, humming_scale_address, donor_bit_position, translate_stage
ROOT=Path(__file__).resolve().parent
DONOR=Path(r'D:/OpenKernelChallenge/muse_moe_w4a16/submission_v139_wgs_kperm3.py')

def raw_code(n,k): return ((n*7) ^ (k*11) ^ (n//8) ^ (k//16)) & 15
def raw_scale(n,g): return ((n*19) ^ (g*37) ^ (n//8)) & 255

def pack_source_block(out,n0,k0,N,transpose):
    # Literal scalar translation of process.cuh loops, distinct from inverse address formula.
    for lane in range(32):
        tmp=[None]*128
        for i in range(8):
            for j in range(8):
                for kk in range(2):
                    ix=[j//2,0,i//2,i%2,j%2,kk]
                    if transpose: ix[3],ix[4]=ix[4],ix[3]
                    flat=ix[0]
                    for d,x in zip((1,4,2,2,2),ix[1:]): flat=flat*d+x
                    tmp[flat]=raw_code(n0+i*8+lane//4,k0+j*8+lane%4*2+kk)
        for outer in range(4):
            for jw in range(4):
                word=0
                for slot in range(8):
                    newk=(slot%4)*2+slot//4
                    word|=tmp[outer*32+jw*8+newk]<<(slot*4)
                off=((k0//16+outer)*2*N+n0//64*128+lane*4+jw)*4
                out[off:off+4]=word.to_bytes(4,'little')

def oracle_stage(unit,stage,gate_up,mul,positions):
    out=bytearray(36864)
    for rb in range(16):
        q,warp=divmod(rb,4)
        for step in range(16):
            grp,oldj=divmod(step,4)
            for g in range(8):
                for oldt in range(4):
                    word=0
                    for reg,(h,quarter) in enumerate(((0,0),(1,0),(0,1),(1,1))):
                        logical=unit*256+rb*16+g+8*h
                        row=(logical//16*8+logical%8+(4096 if logical%16>=8 else 0)) if gate_up else logical
                        for up in range(2):
                            val=raw_code(row,(stage*16+step)*16+8*quarter+2*oldt+up)
                            for b in range(4): word|=((val>>b)&1)<<positions[reg,up,b]
                    off=q*9216+(((grp*4+warp)*32+g*4+oldj)*4+oldt)*4
                    out[off:off+4]=word.to_bytes(4,'little')
                for h in range(2):
                    logical=unit*256+rb*16+g+8*h
                    row=(logical//16*8+logical%8+(4096 if logical%16>=8 else 0)) if gate_up else logical
                    off=q*9216+8192+(((grp*4+warp)*8+g)*4+oldj)*2+h
                    out[off]=raw_scale(row,stage*16+step)
    return out

def main():
    start=time.time(); checks=[]
    doc=ast.parse(DONOR.read_text(encoding='utf-8'))
    ns={'DQ7':True}
    wanted=[n for n in doc.body if isinstance(n,ast.FunctionDef) and n.name in ('_stored_pos','_stored_pos_mul')]
    exec(compile(ast.Module(body=wanted,type_ignores=[]),str(DONOR),'exec'),ns)
    for mul in (False,True):
        pos=ns['_stored_pos_mul' if mul else '_stored_pos']()
        assert len(set(pos.values()))==32
        assert all(donor_bit_position(*key,mul=mul)==value for key,value in pos.items())
    # Every element, both mini-block layouts; independent loop and unique exact bit slot.
    for transpose in (False,True):
        N=128; codes=bytearray(N*128//2)
        for n0 in (0,64):
            for k0 in (0,64): pack_source_block(codes,n0,k0,N,transpose)
        seen=set()
        for n in range(N):
            for k in range(128):
                word,shift=humming_code_address(n,k,N,transpose)
                assert (word,shift) not in seen; seen.add((word,shift))
                got=(int.from_bytes(codes[word*4:word*4+4],'little')>>shift)&15
                assert got==raw_code(n,k),(transpose,n,k,got)
        assert len(seen)==N*128
        checks.append({'kind':'inverse_vs_literal_source_and_bijection','transpose':transpose,'elements':len(seen)})
    for gate_up in (False,True):
        N=8192 if gate_up else 4096
        for unit,stage in ((0,0),(N//256-1,15)):
            codes=bytearray(N*4096//2); scales=bytearray(N*256)
            rows=set()
            for logical in range(unit*256,(unit+1)*256):
                row=(logical//16*8+logical%8+(4096 if logical%16>=8 else 0)) if gate_up else logical
                rows.add(row)
            for n0 in sorted({r//64*64 for r in rows}):
                for k0 in range(stage*256,(stage+1)*256,64): pack_source_block(codes,n0,k0,N,True)
            # Independently reproduce transpose + [0,8,...56,1,9,...57,...] scale permutation.
            for kg in range(stage*16,(stage+1)*16):
                for n0 in sorted({r//64*64 for r in rows}):
                    perm=[x+i for i in range(8) for x in range(0,64,8)]
                    for out_j,src_j in enumerate(perm): scales[kg*N+n0+out_j]=raw_scale(n0+src_j,kg)
            for mul in (False,True):
                got=translate_stage(codes,scales,N,unit,stage,gate_up,mul)
                pos=ns['_stored_pos_mul' if mul else '_stored_pos']()
                expected=oracle_stage(unit,stage,gate_up,mul,pos)
                assert got==expected,(gate_up,unit,stage,mul)
                checks.append({'kind':'full_stage_byte_identity','gate_up':gate_up,'unit':unit,'stage':stage,'mul':mul,'bytes':len(got),'sha256':hashlib.sha256(got).hexdigest()})
    # All 256 scale bit patterns travel untouched; eligibility must not assume a restricted span.
    assert {raw_scale(n,0) for n in range(4096)}==set(range(256))
    result={'status':'PASS_CPU_ADDRESS_ONLY','checks':checks,'all_scale_byte_values_preserved':True,
      'elapsed_seconds':time.time()-start,'cuda_compilation':False,'gpu_execution':False,
      'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('canonical_tile_adapter.py','audit_canonical_tile.py')},
      'donor_sha256':hashlib.sha256(DONOR.read_bytes()).hexdigest()}
    (ROOT/'canonical_tile_cpu_result.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({'status':result['status'],'checks':len(checks),'elapsed_seconds':result['elapsed_seconds']}))
if __name__=='__main__': main()
