from pathlib import Path
import argparse,subprocess,sys
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args()
sys.exit(subprocess.run([sys.executable,str(P/'wait1/offline_wait1_sm90.py'),'--out',args.out,'--count','8']).returncode)
