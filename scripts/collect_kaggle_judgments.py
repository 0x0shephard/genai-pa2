"""Recover saved judge artifacts from Kaggle task logs, then validate coverage."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('log',type=Path)
parser.add_argument('--output',type=Path,default=Path('results/judging'))
args=parser.parse_args()
parts=defaultdict(list)
for line in args.log.read_text().splitlines():
    if line.startswith('PA2_ARTIFACT '):
        _,track,filename,payload=line.split(' ',3)
        if track not in ('hh','ultrafeedback','persona') or filename not in ('run.json','summary.json','judgments.jsonl','attempts.jsonl','failures.jsonl'):
            raise ValueError('Unexpected artifact path in task log')
        parts[track,filename].append(payload)
for track in ('hh','ultrafeedback','persona'):
    if (track,'summary.json') not in parts:
        continue
    artifacts={filename:'\n'.join(parts[track,filename])+'\n' for filename in ('run.json','summary.json','judgments.jsonl','attempts.jsonl','failures.jsonl')}
    try:
        summary=json.loads(artifacts['summary.json'])
        json.loads(artifacts['run.json'])
        judgments=[json.loads(s) for s in artifacts['judgments.jsonl'].splitlines() if s]
        for filename in ('attempts.jsonl','failures.jsonl'):
            for s in artifacts[filename].splitlines():
                if s:json.loads(s)
    except ValueError:
        print(track,'artifact block incomplete; waiting')
        continue
    assert summary['pending']==0, summary
    assert len({r['id'] for r in judgments})==len(judgments)==summary['quality']['judged']
    assert summary['quality']['total']==(40 if track=='persona' else 300)
    destination=args.output/track;destination.mkdir(parents=True,exist_ok=True)
    for filename,content in artifacts.items():
        (destination/filename).write_text(content)
    counts=summary['quality']
    print(f"{track}: {counts['judged']}/{counts['total']} saved; {counts['wins']} wins, {counts['ties']} ties, {counts['losses']} losses; {counts['failures']} failures")
