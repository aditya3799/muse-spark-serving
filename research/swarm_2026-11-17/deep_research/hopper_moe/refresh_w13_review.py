from pathlib import Path
P=Path(__file__).resolve().parent;p=P/'prepare_w13_fused_offline.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("'run() does not explicitly validate int32 contiguous aligned buffers or weight/scale device placement. Current tested fixtures satisfy those constraints.'", "'Root has now added int32, contiguity, and device guards following this review; these preconditions are explicit.'")
p.write_text(s,encoding='utf-8')
