from pathlib import Path
import argparse,subprocess,sys
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args()
sys.exit(subprocess.run([sys.executable,str(P/'sm90_wm48/offline_virtual_k_sm90.py'),'--out',args.out,'--count','1']).returncode)
