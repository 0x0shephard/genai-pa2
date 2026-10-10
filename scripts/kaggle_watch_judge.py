"""Capture the task log, reconnecting when Kaggle's live stream ends early."""
import os
import json
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
cli=str(root/'.venv/bin/kaggle')
log=root/'outputs/kaggle-control/judge-full-current.log'
while True:
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
    with log.open('w') as output:
        proc=subprocess.Popen([cli,'benchmarks','tasks','log','pa2-blind-judge','-m','gemini-3.6-flash'],stdout=output,stderr=subprocess.PIPE,env=env,cwd=root)
        try:
            proc.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            proc.terminate();proc.communicate(timeout=10)
    subprocess.run([sys.executable,'scripts/collect_kaggle_judgments.py',str(log)],cwd=root,check=True)
    summaries=[root/'results/judging'/track/'summary.json' for track in ('hh','ultrafeedback','persona')]
    if all(p.exists() and json.loads(p.read_text())['pending']==0 for p in summaries):
        print('All 640 judgment artifacts saved.',flush=True)
        break
    status=subprocess.run([cli,'benchmarks','tasks','status','pa2-blind-judge'],cwd=root,capture_output=True,text=True,timeout=30,check=True).stdout
    progress=[line for line in log.read_text().splitlines() if line.startswith('Judged ')]
    print(progress[-1] if progress else status,flush=True)
    if re.search(r'gemini-3\.6-flash\s+Completed',status):
        subprocess.run([cli,'benchmarks','tasks','log','pa2-blind-judge','-m','gemini-3.6-flash'],cwd=root,stdout=log.open('w'),check=True)
        subprocess.run([sys.executable,'scripts/collect_kaggle_judgments.py',str(log)],cwd=root,check=True)
        break
