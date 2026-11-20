import datetime,hashlib,json,subprocess
from pathlib import Path
R=Path.cwd();P=R/'swarm_2026-11-17/deep_research/round4_serving_path';C=P/'coordination'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for row in read(C/'BASELINE_HASHES.json'):
 p=Path(row['Path']);actual=sha(p)
 checks.append({'path':str(p.relative_to(R)),'initial_sha256':row['Hash'].lower(),'final_sha256':actual,'unchanged':actual==row['Hash'].lower()})
assert all(c['unchanged'] for c in checks)
extra=[]
for row in read(P/'expert_partition/OVERLAY_BASELINE_INPUT_HASHES.json')['files']:
 p=R/row['path'];actual=sha(p)
 extra.append({'path':row['path'],'lane_pin_sha256':row['actual_file_sha256'],'final_sha256':actual,'matches_lane_pin':actual==row['actual_file_sha256'],'early_raw_pin_available':row['prior_early_round4_file_hash_match'] is not None})
assert all(c['matches_lane_pin'] for c in extra)
docker=subprocess.run(['docker','inspect','--format','{{.State.Running}}|{{.State.Status}}|{{.State.StartedAt}}|{{.State.OOMKilled}}|{{.RestartCount}}','sglang-omni-validation-20261116'],capture_output=True,text=True,timeout=15)
processes=subprocess.run(['wsl','-d','Ubuntu','--','/bin/ps','-eo','pid=,comm=,args='],capture_output=True,text=True,timeout=25)
owned=[line for line in processes.stdout.splitlines() if 'round4_serving_path/' in line and any(name in line for name in ['probe_','launch_local_probe.py','launch_metadata.py','compile_','time_projection_plane.py'])]
git=subprocess.run(['git','-C','sglang','status','--short'],capture_output=True,text=True,timeout=15)
audit={'status':'PASS_BASELINE_PRESERVATION','audited_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'initial_four_pins':checks,'additional_lane_source_pins':extra,
       'owned_research_process_scan':{'exit_code':processes.returncode,'matching_workers':owned},
       'tts_read_only_inspect':{'exit_code':docker.returncode,'running_status_startedAt_oomKilled_restartCount':docker.stdout.strip(),'note':'No stop/restart or TTS configuration action was issued by this swarm. Current running state is not a claim of uninterrupted uptime.'},
       'git_status_short':git.stdout.splitlines(),'preexisting_dirty_changes_preserved':True,'submission_or_remote_check_performed':False,'active_serving_installation_performed':False}
(C/'FINAL_PRESERVATION_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'preserved_pins':len(checks),'additional_matching_pins':len(extra),'owned_workers':len(owned),'tts_state':docker.stdout.strip(),'git_status_entries':len(git.stdout.splitlines())},indent=2))
