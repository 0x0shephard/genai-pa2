import json
from pathlib import Path
import sys
import kaggle_benchmarks as kbench

@kbench.task(name='pa2-blind-judge', description='Private blinded PA2 HH, UltraFeedback and persona evaluation')
def evaluate_alignment(llm) -> dict:
    mount=Path('/kaggle/input')
    print('PA2_INPUT_FILES',[str(p) for p in mount.glob('**/*') if p.is_file()][:50])
    pair_files=list(mount.glob('**/hh.jsonl'))
    assert len(pair_files)==1, pair_files
    inputs=pair_files[0].parent
    archives=list(mount.glob('**/alignment-evaluation.zip'))
    modules=list(mount.glob('**/alignment/judge.py'))
    if archives:
        sys.path.insert(0,str(archives[0]))
    else:
        assert len(modules)==1, modules
        sys.path.insert(0,str(modules[0].parents[1]))
    from alignment.judge import judge_pairs
    from alignment.utils import read_jsonl, write_jsonl
    actual_model=llm.model
    full='gemini-3.6-flash' in actual_model
    print('PA2_JUDGE_MODEL',actual_model,'FULL_EVALUATION',full)
    results={}
    counter=0
    def request(prompt):
        nonlocal counter
        counter+=1
        with kbench.chats.new(f'blind-batch-{counter}'):
            return llm.prompt(prompt)
    for track in ['hh','ultrafeedback','persona']:
        pairs=inputs/f'{track}.jsonl'
        if not full:
            pairs=Path.cwd()/f'smoke-{track}.jsonl'
            write_jsonl(pairs,read_jsonl(inputs/f'{track}.jsonl')[:2])
        destination=Path.cwd()/'outputs'/'judging'/track
        summary=judge_pairs(pairs,destination,judge_model=actual_model,request=request)
        results[track]=summary
        print('PA2_TRACK_SUMMARY',json.dumps(summary))
        for filename in ['judgments.jsonl','failures.jsonl','attempts.jsonl','run.json','summary.json']:
            for line in (destination/filename).read_text().splitlines():
                print('PA2_ARTIFACT',track,filename,line)
        assert summary['pending']==0, summary
        assert summary['quality']['failures']==0, summary
    return results

evaluate_alignment.run(kbench.llm)
