from pathlib import Path
import ast,hashlib,json
p=Path(__file__).resolve().parent;out=p/'serving_humming_v1';rel=Path('python/sglang/srt/layers/muse_spark')
manifest=json.loads((out/'manifest.json').read_text());checks={}
for file in (out/rel).glob('*.py'):
 s=file.read_text(encoding='utf-8-sig');before=ast.dump(ast.parse(s),include_attributes=False)
 if file.name=='muse_spark_nvfp4_moe.py':
  s=s.replace("            c['use_stream_k'] = False", "            # Stream-K reduces partial outputs in BF16 even with FP32 MMA.\n            # Disable it to preserve the stock FP32 reduction contract.\n            c['use_stream_k'] = False")
  s=s.replace("    cache1.mul_(layer.w13_post_scale[safe, None])", "    # GEMM output has already rounded to BF16. Apply the BF16 global scale\n    # in a separate BF16 operation, preserving Marlin's second rounding.\n    cache1.mul_(layer.w13_post_scale[safe, None])")
  s=s.replace("    cache3.mul_(layer.w2_post_scale[safe, None])", "    # Preserve the same BF16-output -> BF16-scale boundary for GEMM2.\n    cache3.mul_(layer.w2_post_scale[safe, None])")
 assert ast.dump(ast.parse(s),include_attributes=False)==before
 file.write_bytes(s.replace('\r\n','\n').encode('utf-8'));assert b'\r' not in file.read_bytes()
 key=str(file.relative_to(out)).replace('\\','/');manifest['files'][key]=hashlib.sha256(file.read_bytes()).hexdigest();checks[key]=True
manifest['encoding']='UTF-8 without BOM, LF';manifest['normalization_ast_unchanged']=checks
(out/'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode('utf-8'))
test=p/'test_constructor.py';s=test.read_text(encoding='utf-8-sig');s=s.replace("file.write_text(text,encoding='utf-8')", "file.write_bytes(text.replace('\\r\\n','\\n').encode('utf-8'))");test.write_bytes(s.encode('utf-8'))
print(json.dumps(manifest,indent=2))
