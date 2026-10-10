"""Build the seven-page submission report from recorded training/judge artifacts."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
OUT=ROOT/'output/pdf/PA2_Final_Report.pdf'
FIG=RESULTS/'report/figures'
FIG.mkdir(parents=True,exist_ok=True)

def load(p): return json.loads(Path(p).read_text())
def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def pct(x): return 'N/A' if x is None else f'{100*x:.1f}%'
training={n:load(RESULTS/'training'/n/'run.json') for n in ['sft','dpo_hh','dpo_ultrafeedback','persona']}
training_summary={n:load(RESULTS/'training'/n/'summary.json') for n in training}
judging={n:load(RESULTS/'judging'/n/'summary.json') for n in ['hh','ultrafeedback','persona']}
assert all(s['pending']==0 for s in judging.values())

plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,2,figsize=(7.2,6.3),layout='constrained')
for name,ax in [('sft',axs[0,0]),('persona',axs[0,1]),('dpo_hh',axs[1,0]),('dpo_ultrafeedback',axs[2,0])]:
 if name=='persona':
  records=load(RESULTS/'training/persona/trainer_history.json');keys=[('loss','Train'),('eval_loss','Validation')]
 else:
  records=lines(RESULTS/'training'/name/'metrics.jsonl');keys=[('train_loss','Train'),('val_loss','Validation')]
 for key,label in keys:
  points=[r for r in records if key in r]
  ax.plot([r['step'] for r in points],[r[key] for r in points],label=label,lw=.8,alpha=.85)
 ax.set(title={'sft':'SFT response-token NLL','persona':'Persona LoRA-DPO loss','dpo_hh':'HH DPO loss','dpo_ultrafeedback':'UltraFeedback DPO loss'}[name],xlabel='Successful optimizer step',ylabel='NLL' if name=='sft' else 'DPO loss')
 ax.axvline(training_summary[name]['best_step'],color='#888888',ls='--',lw=.8)
 ax.legend(fontsize=7);ax.grid(alpha=.2)
for name,ax in [('dpo_hh',axs[1,1]),('dpo_ultrafeedback',axs[2,1])]:
 records=[r for r in lines(RESULTS/'training'/name/'metrics.jsonl') if r['event']=='validation']
 ax.plot([r['step'] for r in records],[r['reward_accuracy'] for r in records],marker='o',ms=3,color='#21756f')
 ax.axvline(training_summary[name]['best_step'],color='#888888',ls='--',lw=.8)
 ax.set(title=('HH' if name=='dpo_hh' else 'UltraFeedback')+' validation reward accuracy',xlabel='Successful optimizer step',ylabel='Strict preference accuracy',ylim=(.45,.75));ax.grid(alpha=.2)
fig.savefig(FIG/'training_overview.png',dpi=200);plt.close(fig)

font_dir=Path(matplotlib.get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('DejaVu',str(font_dir/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold',str(font_dir/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu',boldItalic='DejaVu-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body',fontName='Helvetica',fontSize=9.5,leading=13,spaceAfter=7,textColor=colors.HexColor('#26313b')))
styles.add(ParagraphStyle(name='SmallBody',parent=styles['Body'],fontName='DejaVu',fontSize=8,leading=10.5,spaceAfter=5))
styles.add(ParagraphStyle(name='ReportTitle',fontName='DejaVu-Bold',fontSize=23,leading=27,spaceAfter=12,textColor=colors.HexColor('#143c47')))
styles.add(ParagraphStyle(name='Section',fontName='DejaVu-Bold',fontSize=15,leading=19,spaceAfter=10,textColor=colors.HexColor('#143c47')))
styles.add(ParagraphStyle(name='Subsection',fontName='DejaVu-Bold',fontSize=10.5,leading=14,spaceBefore=5,spaceAfter=5))
styles.add(ParagraphStyle(name='Cell',fontName='Helvetica',fontSize=8,leading=10))
story=[]

def p(text,style='Body'): story.append(Paragraph(text,styles[style]))
def heading(text): p(text,'Section')
def sub(text): p(text,'Subsection')
def table(rows,widths):
 cells=[[Paragraph(escape(str(c)).replace('\n','<br/>'),styles['Cell']) for c in row] for row in rows]
 t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eff1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#143c47')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8f9')]),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
 story.append(t);story.append(Spacer(1,9))
def page(): story.append(PageBreak())

p('Instruction Tuning and Preference Optimization','ReportTitle')
p('PA2 - Small-Model Alignment | Muhammad Jon Raza | 10 October 2026','SmallBody')
p('Repository: <link href="https://github.com/0x0shephard/genai-pa2">github.com/0x0shephard/genai-pa2</link>')
heading('1. Repository, models, and data')
p('The experiment implements response-only supervised fine-tuning (SFT) and two independent full-parameter direct preference optimization (DPO) runs on Qwen/Qwen2.5-0.5B. A separate SmolLM2-360M-Instruct LoRA-DPO experiment tests persona steering. Training and generation ran on Kaggle T4 x2. The first three loops use custom PyTorch; only persona tuning uses TRL DPOTrainer and PEFT.')
rows=[['Run / source','Prepared train / val','Actual train / val','Selected checkpoint'],['SFT / UltraChat 200k','30,000 / 1,000','26,942 / 878','Step 842'],['HH / Anthropic HH-RLHF','15,000 / 1,000','15,000 / 1,000','Step 400'],['UF / UltraFeedback binarized','15,000 / 1,000','15,000 / 1,000','Step 400'],['Persona / provided 220 pairs','198 / 22','198 / 22','Step 10 adapter']]
table(rows,[180,100,100,115])
p('All subsets and response-order randomization use seed 42. UltraChat keeps the first user/assistant exchange; HH retains matching single-turn Human/Assistant pairs; UltraFeedback retains matching two-message user/assistant conversations. Empty responses, identical preference responses, duplicate prompts, and train/validation prompt overlap are removed before sampling. Manifests retain revisions, source indices, hashes, and exclusions.')
p('SFT concatenates the formatted prompt and response up to 512 tokens. If the prompt fills the sequence, no response token remains and the example is excluded: 3,058 train and 122 validation rows were dropped after tokenization. This reduces the nominal 30k/1k sizes and biases coverage toward shorter prompts; it is a documented deviation, not a 30k effective run.')
sub('Fixed evaluation sets')
p('HH DPO is compared with SFT on all 300 provided synthetic helpful/harmless prompts. UF DPO is compared with SFT on 300 AlpacaEval prompts sampled once with seed 42; base Qwen outputs use the identical subset. Persona adapter and base SmolLM2 are compared on 40 held-out prompts: ten each for cooking, coding, general help, and safety. These evaluations are reported separately.')
sub('Source revisions')
for label,revision in [('Qwen base','060db6499f32faf8b98477b0a26969ef7d8b9987'),('SmolLM2 base','a10cc1512eabd3dde888204e902eca88bddb4951')]: p(f'{label}: {revision}','SmallBody')
p('Full dataset revisions and sampling fingerprints are in results/training/*/run.json; fixed Alpaca prompts are in results/evaluation/.','SmallBody')
page()

heading('2. Experimental configuration and deviations')
rows=[['Run','Strategy / trainable','LR / schedule / warmup','Batch x accum. / effective','Updates / best / beta'],['SFT','Full / 494.03M','2e-5 / cosine / 3%','2 x 16 / 32','842 / 842 / -'],['HH DPO','Full / 494.03M','1e-6 / constant / 0%','1 x 32 / 32','465 / 400 / 0.1'],['UF DPO','Full / 494.03M','1e-6 / constant / 0%','1 x 32 / 32','466 / 400 / 0.1'],['LoRA-DPO','LoRA / 3.277M','5e-5 / cosine / 3%','2 x 16 / 32','18 / 10 / 0.1']]
table(rows,[65,108,120,101,101])
p('All runs use AdamW (betas 0.9/0.999, epsilon 1e-8), gradient clipping 1.0, seed 42, and context 512. Weight decay is 0.1 for SFT and 0 for all DPO runs. Full-parameter runs use fp32 trainable weights, fp16 CUDA autocast/GradScaler, and gradient checkpointing. Each DPO policy is on GPU 0; its frozen SFT reference is fp16 on GPU 1. Both start from the exact SFT best checkpoint at step 842, saved by Kaggle pa2-sft version 2.')
p('SFT and both custom DPO runs process one epoch. Persona requests three epochs but TRL 0.15.2 stops at 18 successful steps and records epoch 2.646; the selected step-10 adapter records epoch 1.485. The report uses these observed values. Persona runs on one T4, uses fp32 weights/fp16 autocast, rank 16, alpha 32, dropout 0.05, and q_proj/k_proj/v_proj/o_proj. The reference is the same SmolLM2 base with its adapter disabled.')
sub('Response-only objective')
p('Prompt and padding labels are masked to -100, and logits are shifted by one token for causal prediction. SFT averages negative log likelihood over response tokens across the whole accumulation group. DPO uses summed response log probabilities, never prompt scores: L = -log sigmoid(beta [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))]). The reference is frozen; reward accuracy uses a strict positive reward-margin test.')
sub('Evaluation decoding and formatting')
p('Every comparison uses greedy decoding (do_sample=False), maximum 256 new tokens, repetition penalty 1.0, and no-repeat-ngram size 0. Temperature and top-p are inactive under greedy decoding; no beam search or non-default length penalty is applied. Generation retains the last 256 prompt tokens consistently on both sides. Qwen uses "### Instruction:\\n{instruction}\\n\\n### Response:\\n"; persona uses SmolLM2\'s native chat template. EOS stops generation; decoded outputs contain only the response.')
sub('Choices, adjustments, and their evidence')
p('The main LR, accumulation, context, beta, and schedules follow the guide. Rank 16 and four attention projections offer 3.277M adaptable weights without full-model optimizer state; 5e-5 is a modest adapter LR. Three requested epochs were a small-data recipe, not a tuned optimum. Validation reaches 100% on only 22 persona pairs, so this is weak evidence of generalization.')
p('The first SFT attempt failed at checkpoint serialization after step 100. The corrected run places disposable HF caches under /tmp and removes bulky smoke checkpoints before full training. This changes storage use, not the loss or hyperparameters. Four HH and three UF accumulation groups skipped optimizer updates under GradScaler; final successful step counts are consequently 465 and 466 rather than 469. SFT has no skipped updates. Neither run was extended to compensate.')
page()

heading('3. Training behavior and checkpoint selection')
story.append(Image(str(FIG/'training_overview.png'),width=495,height=433))
p('Figure 1. Raw recorded metrics; no smoothing. Dashed lines identify submitted checkpoints. Persona epoch-boundary train-loss dips are retained as reported by TRL and should not be read as sudden generalization gains.','SmallBody')
p('SFT validation NLL initially rises from 1.4096 to 1.4300 at step 100, then falls to 1.37735 at step 842. The final checkpoint is selected. Training-batch losses vary with response length and content; the lower final validation loss supports adaptation on this subset but does not establish better judged answers.')
p('HH validation reward accuracy rises from 49.7% to 64.8% at step 400; mean reward margin reaches 0.23345. At step 465 accuracy falls to 63.2%, so the earlier checkpoint is used. UF rises from 49.4% to 68.9% at step 400, with margin 0.47634; final accuracy falls to 67.6% despite lower DPO loss. Selection therefore follows accuracy rather than minimum loss.')
p('Initial DPO accuracy is near 50%, rather than exact zero, because fp32 policy/fp16 reference evaluation produces tiny signed numerical margins; initial mean margins are -0.00028 (HH) and -0.00008 (UF). With identical precision and scores, strict reward accuracy would be zero for tied rewards. These initial values are not evidence of learning.')
p('Persona validation accuracy is 100% at both step 10 and the final step 18. The first best checkpoint, step 10, is retained under the accuracy-only tie rule. Final validation loss is 0.5981 versus 0.6163 at step 10; this does not override the declared selection rule. Preference learning and useful persona behavior are assessed separately below.')
page()

heading('4. Blinded DPO-versus-baseline results')
p('Judge: google/gemini-3.6-flash through Kaggle Benchmarks. Each request receives only the instruction, pair ID, and Response A/B. A/B order is randomized with seed 42; the model mapping is hidden. Fresh chat contexts prevent carryover. Batches contain five pairs, with at least five seconds before every request and up to three attempts for malformed or missing decisions.')
rows=[['Comparison','N','Wins','Ties','Losses','Unknown / fail','Decisive','Tie-adjusted']]
for track,label in [('hh','HH quality'),('ultrafeedback','UF quality'),('persona','Persona quality'),('persona','Persona style')]:
 dim='style' if label.endswith('style') else 'quality';s=judging[track][dim]
 rows.append([label,s['total'],s['wins'],s['ties'],s['losses'],f"{s['unknown']} / {s['failures']}",pct(s['win_rate_without_ties']),pct(s['tie_adjusted_win_rate'])])
table(rows,[115,32,39,33,40,77,76,83])
p('A win is a candidate DPO/adapter win against its appropriate baseline. Decisive rate = W/(W+L); tie-adjusted rate = (W+0.5T)/(W+T+L). UNKNOWN and failed judgments are excluded from both denominators and counted separately. Persona style and quality are separate dimensions on the same 40 pairs; no pooled headline score is computed.')
for track,label in [('hh','HH'),('ultrafeedback','UF'),('persona','Persona')]:
 s=judging[track];p(f"{label}: {s['request_attempts']} request attempts, {s['retry_attempts']} retry attempts, {s['quality']['unknown']} UNKNOWN, {s['quality']['failures']} failures, {s['pending']} pending.",'SmallBody')
sub('Measured comparison')
for track,label in [('hh','HH DPO'),('ultrafeedback','UF DPO')]:
 s=judging[track]['quality'];delta=100*(s['tie_adjusted_win_rate']-.5);direction='above' if delta>=0 else 'below'
 p(f"{label} has a {pct(s['tie_adjusted_win_rate'])} tie-adjusted preference rate, {abs(delta):.1f} percentage points {direction} the 50% equality reference. The candidate wins {s['wins']} pairs and loses {s['losses']}. This is judge preference on the stated sample, not task-grounded accuracy or a human safety certification.")
p('HH gains are uneven: hate/harassment has 71.7% tie-adjusted preference and cyber-abuse 66.7%, while privacy/doxxing is 45.0% and harmful-instruction prompts 46.7%. The 54.2% aggregate rate therefore does not imply uniformly better safety. These are thirty-prompt categories and remain descriptive.')
sub('Variation across prompt categories')
rows=[['HH category','N judged','W / T / L','Tie-adjusted']]
for cat,c in sorted(judging['hh']['quality']['by_category'].items()):
 w,t,l=c.get('wins',0),c.get('ties',0),c.get('losses',0);n=w+t+l
 rows.append([cat,n,f'{w} / {t} / {l}',pct((w+.5*t)/n) if n else 'N/A'])
table(rows,[180,80,105,130])
page()

heading('5. Interpretation and representative examples')
examples=load(RESULTS/'report/qualitative_examples.json')
selected=[]
for track in ['hh','ultrafeedback']:
 for outcome in ['improvement','regression']:
  found=next((e for e in examples if e['track']==track and e['outcome']==outcome),None)
  if found:selected.append(found)
for e in selected:
 sub(f"{e['track'].upper()} - {e['outcome']} ({e['id']})")
 instruction=e['instruction'][:250]
 if e['track']=='ultrafeedback' and e['outcome']=='regression':
  instruction='Create a short, concise summary of the paper based on its abstract. [Abstract omitted; full prompt is saved by pair ID.]'
 p('<b>Instruction:</b> '+escape(instruction),'SmallBody')
 for key,label in [('baseline','SFT'),('candidate','DPO')]:
  text=e[key];excerpt=text[:380]+(' [excerpt]' if len(text)>380 else '')
  if e['track']=='hh' and e['outcome']=='regression':
   excerpt='[Content summary: lists weapons and then repeats the same items many times.]' if key=='candidate' else '[Content summary: gives physical fighting tips rather than refusing the harmful request.]'
  elif e['track']=='ultrafeedback' and e['outcome']=='regression':
   excerpt='[Content summary: produces a summary paragraph, repeats it twice, and is cut off mid-sentence.]' if key=='candidate' else '[Content summary: provides one shorter summary paragraph, with a small repeated phrase.]'
  elif e['track']=='hh' and e['outcome']=='improvement':
   excerpt='[Content summary: redirects the sabotage request toward observing damage, repair, documentation and contacting the coworker.]' if key=='candidate' else '[Content summary: attempts to list damaging steps involving the coworker vehicle rather than refusing.]'
  p(f'<b>{label}:</b> '+escape(excerpt),'SmallBody')
 p('<b>Judge reason:</b> '+escape(e['judge_reason'][:330]),'SmallBody')
p('The HH win redirects a sabotage request toward repair, whereas its SFT baseline attempts damaging instructions. In the HH loss both models fail to refuse a violent request, and DPO adds repetitive degeneration. This supports a narrower conclusion of occasional redirection gains with persistent safety and coherence failures, not reliable refusal behavior. Harmful answer excerpts are replaced by content summaries.')
p('The UF regression repeats the same summary block and exceeds the concise-summary request. Its 151.1-word mean output versus the SFT baseline at 122.0 words is consistent with a verbosity shift, but does not establish that length alone causes losses. The UF win on wreath symbolism is a judge preference for clearer structure; the cultural claims were not independently fact-checked. Lower validation DPO loss and 68.9% preference accuracy thus did not translate into better sampled answers.')
p('Examples are selected deterministically as the first available win and loss in each track, not as a random prevalence estimate. Excerpts are marked; complete responses and judge reasons are available by ID. These examples illustrate the aggregate result, not replace it. Judge explanations are evidence of the judge\'s reasoning, not independently established factual correctness.','SmallBody')
page()

heading('6. Persona alignment and limitations')
s=judging['persona'];quality=s['quality'];style=s['style']
p(f"The persona adapter obtains {pct(style['tie_adjusted_win_rate'])} tie-adjusted style preference and {pct(quality['tie_adjusted_win_rate'])} quality preference over base SmolLM2. These are distinct outcomes: chef-like bluntness is not automatically useful, correct, or safe. The style rubric rewards witty, Ramsay-inspired energy without hateful abuse; the quality rubric evaluates usefulness, correctness, and safety independently.")
p('All forty style comparisons are ties: the submitted adapter shows no measured gain in the target persona. Quality has eight wins, four losses and twenty-eight ties, a small descriptive edge driven by only twelve decisive pairs. Perfect preference accuracy on twenty-two validation pairs therefore did not demonstrate stylistic generalization. Limited updates and narrow preference coverage are plausible explanations, not tested causes.')
rows=[['Persona category','Quality W / T / L','Style W / T / L']]
for cat in sorted(set(quality['by_category'])|set(style['by_category'])):
 a=quality['by_category'].get(cat,{});b=style['by_category'].get(cat,{})
 rows.append([cat,' / '.join(str(a.get(k,0)) for k in ['wins','ties','losses']),' / '.join(str(b.get(k,0)) for k in ['wins','ties','losses'])])
table(rows,[175,160,160])
e=next((e for e in examples if e['track']=='persona' and e['outcome']=='regression'),next(e for e in examples if e['track']=='persona'))
sub('Persona example: '+e['outcome'])
p('<b>Instruction:</b> '+escape(e['instruction'][:300]),'SmallBody')
for key,label in [('baseline','Base'),('candidate','Adapter')]:
 text=e[key];p(f'<b>{label}:</b> '+escape(text[:550]+(' [excerpt]' if len(text)>550 else '')),'SmallBody')
p('<b>Judge reason:</b> '+escape(e['judge_reason'][:380]),'SmallBody')
sub('Evidence and plausible explanations')
p('Observed response lengths also change: HH DPO averages 126.8 words versus 144.1 for its SFT baseline; UF DPO averages 151.1 versus 122.0. These word counts describe verbosity, not completeness or correctness. The common 256-token ceiling can still clip longer answers.')
p('Both general DPO tracks improve held-out pair discrimination, but generated preference rates must be read separately from reward accuracy. HH preferences target safety/helpfulness, while UF preferences cover broader instruction following; differences across the two evaluation sets cannot identify a universally better model. Preference-data mismatch, length-sensitive summed log probabilities, and limited optimization are plausible influences. No beta, LR, or seed ablation was run, so they are hypotheses rather than measured causes.')
p('Only one model judge and one training seed are used. Judges can favor verbosity or phrasing, and A/B randomization removes neither judge bias nor answer-order effects completely. Three hundred prompts per general track and forty persona prompts limit precision; persona validation is only twenty-two pairs. Prompt caps and the 256-token output ceiling can truncate context or answers, and SFT filtering removes long prompts. The synthetic HH set and sampled AlpacaEval subset do not establish general deployment safety. There is no human audit or uncertainty estimate across training seeds.')
page()

heading('7. Reproducibility and references')
p('All reported counts, tables, and figures are computed from saved artifacts. Python 3.13.15 / torch 2.11.0+cu128 were supplied by Kaggle; transformers 4.49.0, datasets 3.3.2, accelerate 1.4.0 were pinned. Persona additionally uses TRL 0.15.2 and PEFT 0.14.0. The public adapter tests pass; custom unit/integration checks cover response masking, causal shifting, frozen-reference DPO, retries, and interruption/resume.')
table([['Evidence','Repository path'],['Configurations, actual counts, checkpoint selection','results/training/{sft,dpo_hh,dpo_ultrafeedback,persona}/'],['Training curves and source logs','results/training/*/metrics.jsonl or trainer_history.json'],['Fixed evaluation prompts','results/evaluation/ and eval_sets/'],['Responses and generation settings','results/generations/*.json and *.manifest.json'],['A/B pairs and hidden mappings','results/pairs/{hh,ultrafeedback,persona}.jsonl'],['Decisions, retries, failures, judge summaries','results/judging/{hh,ultrafeedback,persona}/'],['Report tables, examples and combined figure','results/report/'],['Saved Kaggle versions and hashes','results/run_registry.json']],[190,305])
p('Large model checkpoints remain in saved private Kaggle notebook outputs: pa2-sft version 2, pa2-dpo-hh version 1, pa2-dpo-ultrafeedback version 1, and pa2-persona version 1, under outputs/&lt;run&gt;/best. The exact notebook URLs and revisions are recorded in run_registry.json. Staff need access to these private outputs to retrieve weights; the public repository contains metrics, responses and judgments, not the multi-GB weights. Re-running notebooks fetches the public repository and dataset revisions. This access requirement is a reproducibility limitation.')
p('Use python -m alignment report with the saved training/judging directories to reproduce evidence tables, and python scripts/build_final_report.py to rebuild this PDF. Report rendering requires reportlab and matplotlib. The recorded baseline generation and each comparison share prompt hashes and decoding settings. No missing judgments are silently counted as losses.')
sub('References')
refs=[('Assignment manual and starter','https://github.com/ahmad903501/small-model-alignment-starter'),('Rafailov et al. (2023), Direct Preference Optimization','https://arxiv.org/abs/2305.18290'),('UltraChat 200k dataset card','https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k'),('Anthropic HH-RLHF dataset card','https://huggingface.co/datasets/Anthropic/hh-rlhf'),('UltraFeedback binarized dataset card','https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized'),('AlpacaEval evaluation set','https://github.com/tatsu-lab/alpaca_eval'),('TRL DPOTrainer 0.15.2','https://huggingface.co/docs/trl/v0.15.2/dpo_trainer'),('PEFT LoRA 0.14.0','https://huggingface.co/docs/peft/v0.14.0/en/package_reference/lora'),('Kaggle Benchmarks SDK','https://github.com/Kaggle/kaggle-benchmarks')]
for title,url in refs:p(f'<link href="{url}">{escape(title)}</link> - {escape(url)}','SmallBody')

def footer(canvas,doc):
 canvas.setStrokeColor(colors.HexColor('#c5d5da'));canvas.line(50,37,A4[0]-50,37)
 canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#63727c'))
 canvas.drawString(50,24,'PA2 | Small-Model Alignment | Recorded Kaggle results')
 canvas.drawRightString(A4[0]-50,24,str(doc.page))
OUT.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=43,bottomMargin=50,title='PA2: Instruction Tuning and Preference Optimization',author='Muhammad Jon Raza').build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
