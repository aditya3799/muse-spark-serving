from pathlib import Path
import ast
p=Path(__file__).resolve().parent/'offline_sm90_compile.py';s=p.read_text(encoding='utf-8')
s=s.replace('from humming.jit.compiler import NVRTCCompiler','from humming.jit.compiler import NVRTCCompiler\nfrom humming.utils.nvrtc import get_nvrtc_library_path')
s=s.replace("audit=json.loads((P/'sm90_wgmma_metadata_audit.json').read_text(encoding='utf-8'))", "audit_path=P/'sm90_wgmma_metadata_audit.json'\naudit=json.loads(audit_path.read_text(encoding='utf-8'))")
anchor="if a.compile:\n    started=time.time()"
replacement="""if a.compile:
    nvrtc_library=Path(get_nvrtc_library_path())
    library_hash=hashlib.sha256()
    with nvrtc_library.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            library_hash.update(chunk)
    report['compiler_provenance']={
        'nvrtc_signature':NVRTCCompiler.signature(),
        'nvrtc_library_path':str(nvrtc_library),
        'nvrtc_library_sha256':library_hash.hexdigest(),
        'flags':NVRTCCompiler.get_flags('90a',kernel.disable_fast_math),
        'include_dirs':NVRTCCompiler._get_include_dirs(),
        'external_header_limit':'CUDA/libcu++ include paths recorded; full external header closure is not hashed',
        'audit_json_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'launcher_sha256':hashlib.sha256((P/'launch_offline_sm90.py').read_bytes()).hexdigest(),
    }
    assert not torch.cuda.is_initialized()
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    started=time.time()"""
assert anchor in s;s=s.replace(anchor,replacement)
s=s.replace("report.update(status='compiled',compile_seconds", "cmdline=cubin.parent/'cmdline.json'\n        if cmdline.exists():report['compiler_cmdline']=json.loads(cmdline.read_text(encoding='utf-8'))\n        report.update(status='compiled',compile_seconds")
ast.parse(s);p.write_bytes(s.encode('utf-8'));print('offline compiler provenance added')
