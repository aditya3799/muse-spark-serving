from pathlib import Path
import argparse,subprocess,sys
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args()
sys.exit(subprocess.run([sys.executable,str(P/'sm90_heavy_cap/offline_w13_fused_sm90.py'),'--out',args.out,'--count','4']).returncode)
