import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,json,time,hashlib,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
receipt=json.loads((ROOT/'research_2026-11-19/prefill/recovered_public_fc_receipt.json').read_text())
name=receipt['temporary_file'].replace('\\','/');path=pathlib.Path('/mnt/c'+name[2:])
start=time.monotonic();raw=np.memmap(path,dtype='<u2',mode='r');hist=np.zeros(256,dtype=np.int64);digest=hashlib.sha256();chunk=262144
for offset in range(0,len(raw),chunk):
 a=np.asarray(raw[offset:offset+chunk]);hist+=np.bincount((a>>7)&255,minlength=256);digest.update(a.tobytes())
assert digest.hexdigest()==receipt['sha256']
window=np.convolve(hist,np.ones(16,dtype=np.int64),'valid');base=int(np.argmax(window));exceptions=int(len(raw)-window[base]);exact=True
for offset in range(0,len(raw),chunk):
 a=np.asarray(raw[offset:offset+chunk]);e=(a>>7)&255;valid=(e>=base)&(e<base+16)
 codes=(((a>>15)<<11)|(((e-base)&15)<<7)|(a&127)).astype(np.uint16)
 # Real12-bit packing into3bytes per2weights, followed by inverse packing.
 c=codes.astype(np.uint32);pairs=c[::2]|(c[1::2]<<12)
 b0=(pairs&255).astype(np.uint8);b1=((pairs>>8)&255).astype(np.uint8);b2=((pairs>>16)&255).astype(np.uint8)
 back=b0.astype(np.uint32)|(b1.astype(np.uint32)<<8)|(b2.astype(np.uint32)<<16)
 decoded_codes=np.empty_like(codes);decoded_codes[::2]=back&4095;decoded_codes[1::2]=(back>>12)&4095
 out=(((decoded_codes>>11)<<15)|((((decoded_codes>>7)&15)+base)<<7)|(decoded_codes&127)).astype(np.uint16);out[~valid]=a[~valid]
 if not np.array_equal(out,a):exact=False;break
assert exact
weights=int(len(raw));codebytes=weights*12//8
result={'scope':'Read-only actualDFlashFC losslesscodec feasibility, no GEMM/runtime speed claim; no compressed weight artifact retained','source_sha256':digest.hexdigest(),'weights':weights,'source_bytes':weights*2,'best_exponent_window':[base,base+15],'exceptions':exceptions,'exception_fraction':exceptions/weights,'all_actual_words_roundtrip_exact':exact,'packed_bytes':codebytes,'with_6byte_exceptions':codebytes+exceptions*6,'with_8byte_exceptions':codebytes+exceptions*8,'compressed_to_original_ratio_8byte_exceptions':(codebytes+exceptions*8)/(weights*2),'elapsed_seconds':time.monotonic()-start,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'exponent_histogram':hist.tolist()}
(HERE/'compression_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='exponent_histogram'}))
