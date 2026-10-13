"""Build the NeurIPS-format submission report from recorded experiment artifacts.

Requires Tectonic on PATH and requirements-report.txt Python dependencies.
"""
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'
SOURCE=ROOT/'docs/report'
OUTPUT=ROOT/'output/pdf/PA2_Final_Report.pdf'
FIG=DATA/'report/figures'

def load(path): return json.loads(path.read_text())
def lines(path): return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def percent(x): return '--' if x is None else f'{100*x:.1f}\\%'
def escape(s):
    return str(s).replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')
def tabular(filename,columns,header,rows):
    content=[r'\small',r'\begin{tabular}{'+columns+'}',r'\toprule',' & '.join(header)+r' \\',r'\midrule']
    content+=[' & '.join(map(str,row))+r' \\' for row in rows]
    content += [r'\bottomrule',r'\end{tabular}']
    (SOURCE/filename).write_text('\n'.join(content)+'\n')

train={n:load(DATA/'training'/n/'run.json') for n in ['sft','dpo_hh','dpo_ultrafeedback','persona']}
summaries={n:load(DATA/'training'/n/'summary.json') for n in train}
judges={n:load(DATA/'judging'/n/'summary.json') for n in ['hh','ultrafeedback','persona']}
assert all(j['pending']==0 for j in judges.values())
(SOURCE/'metrics.tex').write_text('% Tables below are generated from saved run artifacts.\n')
names={'sft':'SFT','dpo_hh':'HH DPO','dpo_ultrafeedback':'UF DPO','persona':'Persona LoRA-DPO'}
tabular('data_table.tex','lrrr',['Run','Prepared train/val','Effective train/val','Best step'],[
 [names[n], '30,000/1,000' if n=='sft' else '198/22' if n=='persona' else '15,000/1,000',f"{train[n]['actual_examples']['train']:,}/{train[n]['actual_examples']['validation']:,}",summaries[n]['best_step']] for n in train])
tabular('config_table.tex','llllrrr',['Run','Strategy','LR; schedule','Batch $\\times$ accum.','Updates','Best','$\\beta$'],[
 [names[n],'LoRA' if n=='persona' else 'Full',('$5\\times10^{-5}$; cosine' if n=='persona' else '$2\\times10^{-5}$; cosine' if n=='sft' else '$10^{-6}$; constant'),f"{train[n]['config']['batch_size']} $\\times$ {train[n]['config']['accumulation']}",summaries[n]['step'],summaries[n]['best_step'],train[n]['config'].get('beta','--')] for n in train])
rows=[]
for track,dim,label in [('hh','quality','HH quality'),('ultrafeedback','quality','UF quality'),('persona','quality','Persona quality'),('persona','style','Persona style')]:
    j=judges[track][dim];rows.append([label,j['total'],j['wins'],j['ties'],j['losses'],percent(j['win_rate_without_ties']),percent(j['tie_adjusted_win_rate'])])
tabular('results_table.tex','lrrrrrr',['Comparison','$N$','Wins','Ties','Losses','Decisive','Tie-adjusted'],rows)
rows=[]
for cat,c in sorted(judges['hh']['quality']['by_category'].items()):
    w,t,l=(c.get(k,0) for k in ['wins','ties','losses']);rows.append([escape(cat.replace('_',' ')),w,t,l,percent((w+.5*t)/(w+t+l))])
tabular('category_table.tex','lrrrr',['Category','Wins','Ties','Losses','Tie-adjusted'],rows)
rows=[]
for cat,c in sorted(judges['persona']['quality']['by_category'].items()):
    st=judges['persona']['style']['by_category'][cat]
    rows.append([cat.replace('_',' '),' / '.join(str(c.get(k,0)) for k in ['wins','ties','losses']),' / '.join(str(st.get(k,0)) for k in ['wins','ties','losses'])])
tabular('persona_table.tex','lcc',['Category','Quality: W / T / L','Style: W / T / L'],rows)

plt.rcParams.update({'font.size':10,'text.color':'black','axes.labelcolor':'black','axes.edgecolor':'black','xtick.color':'black','ytick.color':'black','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,2,figsize=(7.2,6.6),layout='constrained')
for n,ax in [('sft',axes[0,0]),('persona',axes[0,1]),('dpo_hh',axes[1,0]),('dpo_ultrafeedback',axes[2,0])]:
    records=load(DATA/'training/persona/trainer_history.json') if n=='persona' else lines(DATA/'training'/n/'metrics.jsonl')
    keys=[('loss','Training','-'),('eval_loss','Validation','--')] if n=='persona' else [('train_loss','Training','-'),('val_loss','Validation','--')]
    for key,label,style in keys:
        points=[r for r in records if key in r];ax.plot([r['step'] for r in points],[r[key] for r in points],color='black',ls=style,label=label,lw=.8)
    ax.set(title={'sft':'SFT loss','persona':'Persona LoRA-DPO loss','dpo_hh':'HH DPO loss','dpo_ultrafeedback':'UF DPO loss'}[n],xlabel='Optimizer step',ylabel='NLL' if n=='sft' else 'DPO loss')
    ax.axvline(summaries[n]['best_step'],color='black',ls=':',lw=.8);ax.legend(fontsize=8,frameon=False)
for n,ax in [('dpo_hh',axes[1,1]),('dpo_ultrafeedback',axes[2,1])]:
    records=[r for r in lines(DATA/'training'/n/'metrics.jsonl') if r['event']=='validation']
    ax.plot([r['step'] for r in records],[r['reward_accuracy'] for r in records],color='black',marker='o',ms=3)
    ax.axvline(summaries[n]['best_step'],color='black',ls=':',lw=.8)
    ax.set(title=('HH' if n=='dpo_hh' else 'UF')+' validation accuracy',xlabel='Optimizer step',ylabel='Reward accuracy',ylim=(.45,.75))
FIG.mkdir(parents=True,exist_ok=True)
fig.savefig(FIG/'training_overview.png',dpi=220);plt.close(fig)
compiler=shutil.which('tectonic')
if compiler is None: raise RuntimeError('Install Tectonic to compile the official NeurIPS LaTeX template.')
build=ROOT/'tmp/pdfs/neurips-build';build.mkdir(parents=True,exist_ok=True)
subprocess.run([compiler,'--keep-logs','--outdir',str(build),'report.tex'],cwd=SOURCE,check=True)
OUTPUT.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(build/'report.pdf',OUTPUT)
from pypdf import PdfReader
pages=len(PdfReader(OUTPUT).pages)
assert pages<=8,f'Report has {pages} pages; assignment permits eight including references.'
print(f'{OUTPUT} ({pages} pages)')
