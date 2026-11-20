from pathlib import Path
import hashlib
p=Path(__file__).resolve().parent/'probe_swap_full_helper.py';s=p.read_text()
a="for pattern in ('uniform','hot','mixed'):";assert s.count(a)==1;s=s.replace(a,"for pattern in ('uniform','hot','mixed','all_remote'):")
a="P.parent/'aux_speculation/adversarial_fixture.py',CONFIG]";assert s.count(a)==1;s=s.replace(a,"P.parent/'aux_speculation/adversarial_fixture.py',CONFIG,P/'SWAP_FULL_HELPER_PREREQUISITES.json',*sorted((OLD/'humming_vendor/humming').rglob('*.py')),*sorted((OLD/'humming_vendor/humming/include').rglob('*.cuh'))]")
p.write_text(s);compile(s,str(p),'exec');print(hashlib.sha256(p.read_bytes()).hexdigest())
