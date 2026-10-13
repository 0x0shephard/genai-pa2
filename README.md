# PA2: small-model SFT and DPO

Complete training/evaluation code for the assignment in [PA2.pdf](PA2.pdf). Training runs on **Kaggle GPU T4 x2**; the local grading tests need only PyTorch and pytest.


## Completed assignment results

The Kaggle experiments and all 640 blinded Gemini 3.6 Flash comparisons are complete, with zero failed or unknown judgments. The final [seven-page PDF report](output/pdf/PA2_Final_Report.pdf) uses the official NeurIPS 2026 format with black text and monochrome plots, and contains actual configurations, results, examples, and limitations.

| Comparison | Wins / ties / losses | Tie-adjusted preference |
|---|---|---|
| HH DPO vs SFT | 128 / 69 / 103 | 54.2% |
| UltraFeedback DPO vs SFT | 99 / 72 / 129 | 45.0% |
| Persona adapter vs base: quality | 8 / 28 / 4 | 55.0% |
| Persona adapter vs base: style | 0 / 40 / 0 | 50.0% (all ties) |

HH shows a small aggregate preference gain; UF favors SFT; the persona run shows no measured style gain. These are descriptive model-judge results from one seed. SFT effectively trained on 26,942/878 examples after response-span filtering, and persona training stopped at 18 updates (recorded epoch 2.646); the report documents these deviations.

Evidence is under [results/](results/), with exact saved Kaggle versions in [run_registry.json](results/run_registry.json). Large checkpoints remain in private Kaggle notebook outputs. The report and evidence are also saved in your [private Kaggle submission dataset](https://www.kaggle.com/datasets/jonraza15/pa2-completed-assignment).

The editable LaTeX source is in [docs/report/](docs/report/). To rebuild the PDF locally, make sure Tectonic is available on PATH:

```bash
python -m pip install -r requirements-report.txt
python scripts/build_final_report.py
```

## Start here

1. Publish this directory to your **public GitHub repository**. You manage the remote and push yourself.
2. Upload [01_sft.ipynb](notebooks/01_sft.ipynb) to a Kaggle notebook. Enable **Internet** and **GPU T4 x2**. Enter your GitHub URL in `REPO_URL`.
3. Run the smoke cells. Inspect losses and five sample responses. Set `RUN_FULL=True`, then run the full cells. `RUN_FULL=False` intentionally does not launch final assignment training.
4. Save a Kaggle version with outputs. Confirm `outputs/sft/best`, `outputs/sft/final`, `outputs/sft/resume.pt`, and `prepared/` are present in the saved version.
5. Copy the printed Git commit into `GIT_REF` in subsequent notebooks so every submitted stage runs the same code.

For the required base-model baseline, run [04_generate.ipynb](notebooks/04_generate.ipynb) with `MODE="base"` before SFT. The same notebook later generates trained-model outputs.

## Notebook workflow

| Notebook | How to run | Save and attach |
|---|---|---|
| [01 SFT](notebooks/01_sft.ipynb) | One final run | SFT checkpoints, prepared subsets, metrics |
| [02 DPO](notebooks/02_dpo.ipynb) | Twice: `TRACK="hh"`, then `TRACK="ultrafeedback"` | Attach the **same SFT best** to each independent run |
| [03 Persona](notebooks/03_persona_lora.ipynb) | One final LoRA-DPO run | Adapter checkpoints and training history |
| [04 Generation](notebooks/04_generate.ipynb) | First base-only; later `MODE="trained"` | Attach all trained checkpoints and fixed Alpaca prompts |
| [05 Judge/report](notebooks/05_judge_report.ipynb) | Benchmark-enabled notebook, no GPU needed | Attach pairs and training runs; saves judgments/report evidence |

Use **Add Input → Notebook Output** to attach saved notebook versions. Then edit each notebook's `/kaggle/input/YOUR-…` paths. Input dataset slugs depend on your account; notebook errors list available attachments. SFT/DPO checkpoints are directories, not single `.pt` weight files.

The judging notebook must run in a benchmark/model-proxy environment. Start at <https://www.kaggle.com/benchmarks/tasks/new>, then import the notebook or paste its cells. The SDK is already provided there; a normal GPU notebook with a pip-installed SDK alone may lack model authorization. If the configured judge is unavailable, inspect `kbench.llms`, choose an available judge, and record the change.

Notebook settings contain a public clone URL only. They do not push to GitHub or store credentials. No local GPU training is required.

## Experiment defaults

- **SFT:** Qwen/Qwen2.5-0.5B; UltraChat first user/assistant exchange; 30,000/1,000 examples; AdamW; LR 2e-5; 3% warmup and cosine schedule; weight decay 0.1; batch 2 × accumulation 16; one epoch.
- **Each DPO:** same SFT best as policy and reference; 15,000/1,000 single-turn pairs; AdamW; fixed LR 1e-6; beta 0.1; no weight decay; batch 1 × accumulation 32; one epoch. Policy on GPU 0, frozen reference on GPU 1.
- **Persona:** SmolLM2-360M-Instruct; seeded 90/10 split of provided data; TRL DPOTrainer; LoRA r=16, alpha=32, dropout=0.05 on q/k/v/o projections; LR 5e-5; 3% warmup/cosine; batch 2 × accumulation 16; three epochs. Trainer is restricted to one visible GPU to preserve the declared effective batch size.
- All tracks: seed 42, context 512, gradient clip 1.0, fp32 trainable weights, fp16 CUDA autocast. DPO prompt cap 256. Best/final checkpoints are saved separately. Validation every 100 optimizer steps and at epoch end; persona every 10 steps.
- Qwen uses `### Instruction:\n{instruction}\n\n### Response:\n`; persona uses its model's chat template. Qwen prompt/response tokens are encoded separately, without automatic special tokens, to preserve an explicit masking boundary. EOS is appended to response tokens before truncation.
- Generation: greedy, maximum 256 new tokens, 256-token prompt cap, no repetition penalty by default. SFT and DPO use identical settings in each comparison.

The normalized subsets and manifests record dataset revisions, filtering, IDs, hashes, and counts. Training logs also record examples dropped for lacking scoreable response tokens after truncation. Report these actual counts, rather than claiming every sampled row was optimized.

## Code and grading interface

`alignment/core.py` implements response-only SFT, causal shifting, summed response log probabilities, and DPO math. `alignment/train.py` implements both main training loops directly with PyTorch. It does not import Trainer, TRL, PEFT, or a prebuilt DPO loop. Only `alignment/persona.py` uses TRL/PEFT.

`submission_adapter.py` preserves the six required signatures and delegates to this same core. Adapter prompts are **already formatted strings**. SFT reduction is the mean over scoreable response tokens, including EOS when retained. Log-probability sums exclude prompt and padding tokens. Batches with entirely truncated responses have zero response log-probability; a loss with no response tokens is rejected. Training filters such rows and logs their IDs.

Padding uses an attention mask, even when pad ID equals EOS ID. Main DPO uses strict `reward_chosen > reward_rejected`, not a tie score of 0.5. Identical log probabilities yield loss `log(2)` and accuracy zero. An fp16 reference can produce tiny numerical differences at initialization.

## Resume and storage

- **Main training:** set `RESUME_STATE` to a saved run's `resume.pt`. Attach its sibling `best/` directory too. Resume restores model, optimizer, scheduler, scaler, random states, metrics, epoch, and the next accumulation group. Configuration/data fingerprints must match. A smoke run cannot be resumed as a full run.
- **Persona:** set `RESUME_CHECKPOINT` to a saved `trainer/checkpoint-N` directory. Trainer restores its checkpoint. Use the same data, model revision, and configuration.
- **Generation:** set `GENERATION_RESUME_DIR` to the saved `outputs/generations` directory. Keep each JSON's sibling `.manifest.json`. Completed prompt IDs are skipped only when the manifest matches.
- **Judging:** set `JUDGE_RESUME_ROOT` to saved `outputs/judging`. Completed decisions are skipped; unresolved failures are retried. Keep `run.json`, `judgments.jsonl`, `failures.jsonl`, and `attempts.jsonl` together.

Metrics, generations, and judgments use atomic writes. A single replaceable `resume.pt` is saved at validation boundaries; resume may repeat work since the last boundary. `best/`, `final/`, and resume state can occupy roughly 10 GB for a full Qwen run; do not duplicate them into archives on limited Kaggle disk. Save each DPO track in a separate Kaggle version.

Checkpoints written into a live session do not survive session deletion unless Kaggle preserves the notebook outputs. Saving a notebook version is part of completing each stage. Hugging Face download caches need not be carried to later stages.

## Evaluation and report

Required comparisons: HH DPO versus SFT on the provided 300 HH prompts; UltraFeedback DPO versus SFT on 300 AlpacaEval prompts selected with seed 42; persona LoRA versus unmodified SmolLM2 on 40 fixed prompts. Persona style and usefulness/safety are judged separately.

Pair files randomize response order and store identity mappings under `hidden_metadata`. The judge-facing prompt contains only IDs, instructions, and A/B responses. Do not feed entire pair records directly to a judge.

Judging uses batches of five, a five-second delay before **every** attempt, up to three attempts per batch, incremental checkpoints, and failure logging. Valid decisions from partial batches are retained; missing/invalid decisions are retried. `UNKNOWN` is distinct from a failed call. Summary denominators are documented in generated results.

The judge notebook creates figures and report evidence. The completed submission PDF is linked above; scripts/build_final_report.py rebuilds it from the recorded results. Intermediate large checkpoints and control files under outputs/ are ignored.

## Offline checks

In an environment with PyTorch installed:

```bash
python -m pip install pytest nbformat
PYTHONPATH=. python -m pytest -q
python scripts/validate_notebooks.py
```

Kaggle notebooks run the public smoke tests before GPU work. Extra tests cover numerical causal scoring, accumulation, checkpoint restoration, truncation, dataset filtering, blind comparisons, retry/resume, and reporting denominators. CPU tests use tiny models and mocked judge calls; they do not download pretrained models or perform assignment training.

Rebuild notebooks after editing their source generator:

```bash
python scripts/build_notebooks.py
```

CLI help: `python -m alignment --help`; each subcommand has `--help`. Dependencies are intentionally separate from the editable package install so Kaggle's PyTorch is retained.

## Attribution

Fixed persona/HH data and public tests are copied unchanged from [the course starter](https://github.com/ahmad903501/small-model-alignment-starter). The implementation is written for this project. External dataset and API references are listed in [the report template](docs/report_template.md). Follow the manual's academic integrity and citation requirements.
