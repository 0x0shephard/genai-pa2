"""Deterministic notebook authoring; no execution or model downloads."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": textwrap.dedent(text).strip() + "\n"}


CONFIG = '''
from pathlib import Path
import json, os, subprocess, sys

REPO_URL = "https://github.com/0x0shephard/genai-pa2.git"
GIT_REF = "main"  # After the first run, use its printed commit SHA for all final runs.
REPO_DIR = Path("/kaggle/working/pa2-repo")
OUTPUT_ROOT = Path("/kaggle/working/outputs")
PREPARED_ROOT = Path("/kaggle/working/prepared")
RUN_SMOKE = True
RUN_FULL = False  # Inspect smoke output, then set True and rerun the configuration/full cells.
'''

SETUP = '''
assert "YOUR_" not in REPO_URL, "Enter your public GitHub repository URL in the configuration cell."
assert REPO_URL.startswith("https://github.com/"), "Use a public GitHub HTTPS URL."
REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
if not (REPO_DIR / ".git").exists():
    subprocess.check_call(["git", "clone", REPO_URL, str(REPO_DIR)])
else:
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=REPO_DIR, text=True).strip()
    assert origin.rstrip("/").removesuffix(".git") == REPO_URL.rstrip("/").removesuffix(".git"), "Existing checkout belongs to another repository."
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_DIR, text=True).strip(), "Checkout has local changes; preserve them before changing revisions."
subprocess.check_call(["git", "fetch", "origin", GIT_REF], cwd=REPO_DIR)
subprocess.check_call(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO_DIR)
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()
os.environ["ALIGNMENT_GIT_REF"] = GIT_REF
os.environ["PYTHONPATH"] = str(REPO_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/kaggle/working/hf-cache"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
print("Git commit:", COMMIT)

def run(*args):
    subprocess.check_call([sys.executable, "-m", "alignment", *map(str, args)], cwd=REPO_DIR)

def need(path, label):
    path = Path(path)
    if not path.exists():
        print("Available attachments:", list(Path("/kaggle/input").glob("*/*")))
        raise FileNotFoundError(f"Attach the saved {label} output and update its configuration path: {path}")
    return path

def completed(path):
    summary = Path(path) / "summary.json"
    return summary.exists() and json.loads(summary.read_text()).get("completed", False)

def show_samples(path, count=5):
    for row in json.loads(Path(path).read_text())[:count]:
        print("PROMPT:", row["instruction"])
        print("RESPONSE:", row["output"], "\\n")
'''


def install(persona=False, judge=False):
    if judge:
        return code('''
        # Keep the benchmark runtime and its model-proxy dependencies intact.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml", "matplotlib"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR), "--no-deps"])
        sys.path.insert(0, str(REPO_DIR))
        import kaggle_benchmarks as kbench
        print("Kaggle model registry:", kbench.llms)
        ''')
    requirements = "requirements-persona.txt" if persona else "requirements.txt"
    return code(f'''
    # Keep Kaggle's installed CUDA PyTorch. These requirements do not pin torch.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REPO_DIR / "{requirements}"), "--upgrade-strategy", "only-if-needed"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR), "--no-deps"])
    subprocess.check_call(["nvidia-smi"])
    subprocess.check_call([sys.executable, "-c", "import torch, transformers, datasets; print('torch', torch.__version__, 'transformers', transformers.__version__, 'datasets', datasets.__version__); print('GPU count:', torch.cuda.device_count()); assert torch.cuda.device_count() >= 2, 'Enable GPU T4 x2'"])
    subprocess.check_call([sys.executable, "-m", "pytest", "-q", "tests/test_public_adapter_smoke.py"], cwd=REPO_DIR)
    ''')


def shared(title, intro, extra="", persona=False, judge=False):
    return [markdown(f"# {title}\n\n{intro}\n\nUpload this notebook to Kaggle. Enable Internet; select **GPU T4 x2** for training/generation. The judge notebook needs benchmark model access, not a GPU. Enter your public GitHub URL below. No GitHub credentials are needed."),
            markdown("## 1. Configuration\n\nRun All defaults to smoke checks only. After inspecting them, set `RUN_FULL=True`. Use a new run name if changing experimental settings; completed runs are skipped."),
            code(CONFIG + "\n" + textwrap.dedent(extra).strip()),
            markdown("## 2. Clone the chosen GitHub revision and install dependencies"), code(SETUP), install(persona, judge)]


SAVE = markdown('''
## Preserve outputs

Use **Save Version → Save & Run All** after setting `RUN_FULL=True`, or save completed interactive outputs with the appropriate Kaggle version option. Attach that saved notebook's output to the next stage using **Add Input → Notebook Output**. Confirm the files exist in the saved version before ending the session.

New artifacts are under `/kaggle/working/outputs` and `/kaggle/working/prepared`. Model files are large: do not duplicate them into zip archives unless disk permits. For interruption recovery, attach the prior saved output and configure its resume path. A file in a temporary session is preserved only when Kaggle saves the output.
''')


def write(name, cells, gpu=True):
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"},
                             "kaggle": {"isInternetEnabled": True, "isGpuEnabled": gpu}}, "cells": cells}
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:02d}"
    path = ROOT / "notebooks" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2) + "\n")


def main():
    cells = shared("01 · SFT", "Train Qwen2.5-0.5B with your own response-only PyTorch loop.", '''
    RUN_NAME = "sft"
    RESUME_STATE = None  # e.g. /kaggle/input/saved-sft/outputs/sft/resume.pt
    ''')
    cells += [markdown("## 3. Small training smoke test\n\nThe pilot uses 64/8 examples and two optimizer steps in its own directory. It never replaces the final run."), code('''
    if RUN_SMOKE:
        smoke_data = PREPARED_ROOT / "smoke"
        smoke_run = OUTPUT_ROOT / "smoke_sft"
        run("prepare", "--track", "sft", "--output", smoke_data, "--train-size", 64, "--val-size", 8)
        if not completed(smoke_run):
            run("sft", "--data", smoke_data / "sft", "--output", smoke_run, "--max-steps", 2, "--eval-steps", 1)
        run("generate", "--checkpoint", smoke_run / "best", "--prompts", REPO_DIR / "eval_sets/hh_rlhf_eval_300_seed42.json",
            "--output", OUTPUT_ROOT / "sanity/sft_smoke.json", "--generator", "sft-smoke", "--limit", 5)
        show_samples(OUTPUT_ROOT / "sanity/sft_smoke.json")
    '''), markdown("## 4. Full SFT training\n\nDefaults: 30k/1k examples, seed 42, context 512, batch 2 × accumulation 16, LR 2e-5, one epoch. Review smoke losses and response behavior first."), code('''
    if RUN_FULL:
        run("prepare", "--track", "sft", "--output", PREPARED_ROOT)
        run("prepare", "--track", "alpaca", "--output", PREPARED_ROOT / "eval")
        destination = OUTPUT_ROOT / RUN_NAME
        if not completed(destination):
            args = ["sft", "--data", PREPARED_ROOT / "sft", "--output", destination]
            if RESUME_STATE:
                args += ["--resume", need(RESUME_STATE, "SFT resume state")]
            run(*args)
        run("plot", "--run", destination, "--output", destination / "training.png", "--name", "SFT")
    else:
        print("Smoke only. Inspect results, then set RUN_FULL=True for assignment training.")
    '''), SAVE]
    write("01_sft.ipynb", cells)

    cells = shared("02 · From-scratch DPO", "Run this notebook twice, independently from the same SFT best checkpoint: once for HH, once for UltraFeedback.", '''
    TRACK = "hh"  # Change to "ultrafeedback" for the second independent run.
    SFT_BEST = Path("/kaggle/input/YOUR-SFT-OUTPUT/outputs/sft/best")
    RUN_NAME = f"dpo_{TRACK}"
    RESUME_STATE = None  # Saved outputs/.../resume.pt, with its sibling best directory.
    ''')
    cells += [markdown("## 3. Verify SFT input and run a pilot"), code('''
    assert TRACK in ("hh", "ultrafeedback")
    need(SFT_BEST / "alignment_manifest.json", "SFT best checkpoint")
    if RUN_SMOKE:
        smoke_data = PREPARED_ROOT / "smoke"
        smoke_run = OUTPUT_ROOT / f"smoke_dpo_{TRACK}"
        run("prepare", "--track", TRACK, "--output", smoke_data, "--train-size", 64, "--val-size", 8)
        if not completed(smoke_run):
            run("dpo", "--data", smoke_data / TRACK, "--model", SFT_BEST, "--output", smoke_run,
                "--max-steps", 2, "--eval-steps", 1)
        print((smoke_run / "metrics.jsonl").read_text())
    '''), markdown("## 4. Full independent DPO run\n\n15k/1k pairs, beta 0.1, fixed LR 1e-6, batch 1 × accumulation 32, one epoch. Policy uses GPU 0; frozen SFT reference uses GPU 1."), code('''
    if RUN_FULL:
        run("prepare", "--track", TRACK, "--output", PREPARED_ROOT)
        destination = OUTPUT_ROOT / RUN_NAME
        if not completed(destination):
            args = ["dpo", "--data", PREPARED_ROOT / TRACK, "--model", SFT_BEST, "--output", destination]
            if RESUME_STATE:
                args += ["--resume", need(RESUME_STATE, "DPO resume state")]
            run(*args)
        run("plot", "--run", destination, "--output", destination / "training.png", "--name", RUN_NAME)
    else:
        print("Smoke only. Set RUN_FULL=True after inspecting the pilot.")
    '''), SAVE]
    write("02_dpo.ipynb", cells)

    cells = shared("03 · Persona LoRA-DPO", "Use TRL DPOTrainer and PEFT for the provided persona preferences. The CLI restricts Trainer to one GPU to keep effective batch size 32.", '''
    RUN_NAME = "persona"
    RESUME_CHECKPOINT = None  # e.g. /kaggle/input/saved-persona/outputs/persona/trainer/checkpoint-20
    ''', persona=True)
    cells += [markdown("## 3. Prepare the fixed split and smoke-test TRL"), code('''
    run("prepare", "--track", "persona", "--output", PREPARED_ROOT)
    if RUN_SMOKE:
        smoke_run = OUTPUT_ROOT / "smoke_persona"
        if not completed(smoke_run):
            run("persona", "--data", PREPARED_ROOT / "persona", "--output", smoke_run, "--max-steps", 2)
        print((smoke_run / "summary.json").read_text())
    '''), markdown("## 4. Full persona run\n\nRank 16, alpha 32, attention projections, dropout 0.05, LR 5e-5, three epochs. See the report template for the rationale to verify against actual results."), code('''
    if RUN_FULL:
        destination = OUTPUT_ROOT / RUN_NAME
        if not completed(destination):
            args = ["persona", "--data", PREPARED_ROOT / "persona", "--output", destination]
            if RESUME_CHECKPOINT:
                args += ["--resume", need(RESUME_CHECKPOINT, "TRL resume checkpoint")]
            run(*args)
        run("plot", "--run", destination, "--output", destination / "training.png", "--name", "Persona LoRA-DPO")
    else:
        print("Smoke only. Set RUN_FULL=True for the final persona run.")
    '''), SAVE]
    write("03_persona_lora.ipynb", cells)

    cells = shared("04 · Generation and comparisons", "Run MODE='base' before training for baseline outputs. After training, attach checkpoint outputs and run MODE='trained'. Generations persist after each prompt.", '''
    MODE = "base"  # "base" or "trained"
    SFT_BEST = Path("/kaggle/input/YOUR-SFT-OUTPUT/outputs/sft/best")
    HH_BEST = Path("/kaggle/input/YOUR-HH-OUTPUT/outputs/dpo_hh/best")
    UF_BEST = Path("/kaggle/input/YOUR-UF-OUTPUT/outputs/dpo_ultrafeedback/best")
    PERSONA_BEST = Path("/kaggle/input/YOUR-PERSONA-OUTPUT/outputs/persona/best")
    # In trained mode, attach SFT prepared prompts to preserve the exact Alpaca subset.
    ALPACA_PROMPTS = None  # e.g. /kaggle/input/YOUR-SFT-OUTPUT/prepared/eval/alpaca_eval_300_seed42.json
    GENERATION_RESUME_DIR = None  # A saved output directory containing generation JSONs and manifests.
    MAX_NEW_TOKENS = 256
    REPETITION_PENALTY = 1.0
    NO_REPEAT_NGRAM_SIZE = 0
    ''', persona=True)
    cells += [markdown("## 3. Fix evaluation prompts and model jobs"), code('''
    assert MODE in ("base", "trained")
    if ALPACA_PROMPTS:
        alpaca = need(ALPACA_PROMPTS, "fixed AlpacaEval prompts")
    else:
        run("prepare", "--track", "alpaca", "--output", PREPARED_ROOT / "eval")
        alpaca = PREPARED_ROOT / "eval/alpaca_eval_300_seed42.json"
    hh = REPO_DIR / "eval_sets/hh_rlhf_eval_300_seed42.json"
    persona_prompts = REPO_DIR / "eval_sets/persona_eval_40_seed42.json"
    jobs = [("qwen2.5-0.5b-base", "Qwen/Qwen2.5-0.5B", alpaca, False)] if MODE == "base" else [
        ("sft_alpaca", SFT_BEST, alpaca, False), ("dpo_ultrafeedback", UF_BEST, alpaca, False),
        ("sft_hh", SFT_BEST, hh, False), ("dpo_hh", HH_BEST, hh, False),
        ("smollm2-base", "HuggingFaceTB/SmolLM2-360M-Instruct", persona_prompts, True),
        ("persona_lora", PERSONA_BEST, persona_prompts, True)]
    for _, checkpoint, _, _ in jobs:
        if isinstance(checkpoint, Path):
            need(checkpoint / "alignment_manifest.json", "trained checkpoint")

    def generate_job(name, checkpoint, prompts, is_persona, smoke=False):
        folder = OUTPUT_ROOT / ("sanity" if smoke else "generations")
        args = ["generate", "--checkpoint", checkpoint, "--prompts", prompts,
                "--output", folder / f"{name}.json", "--generator", name,
                "--max-new-tokens", MAX_NEW_TOKENS, "--repetition-penalty", REPETITION_PENALTY,
                "--no-repeat-ngram-size", NO_REPEAT_NGRAM_SIZE]
        if is_persona:
            args += ["--persona"]
        if smoke:
            args += ["--limit", 5]
        elif GENERATION_RESUME_DIR and (Path(GENERATION_RESUME_DIR) / f"{name}.json").exists():
            args += ["--resume-from", Path(GENERATION_RESUME_DIR) / f"{name}.json"]
        run(*args)
        return folder / f"{name}.json"
    '''), markdown("## 4. Inspect five responses per model\n\nConfirm paths, prompt formatting, EOS, length, and repetition before launching the full sets. If decoding changes are needed, use a new output directory and apply them to both sides."), code('''
    if RUN_SMOKE:
        for job in jobs:
            show_samples(generate_job(*job, smoke=True))
    '''), markdown("## 5. Full generation and blind comparison files"), code('''
    if RUN_FULL:
        for job in jobs:
            generate_job(*job)
        if MODE == "trained":
            generations = OUTPUT_ROOT / "generations"
            for track, baseline, candidate in [("hh", "sft_hh", "dpo_hh"),
                ("ultrafeedback", "sft_alpaca", "dpo_ultrafeedback"),
                ("persona", "smollm2-base", "persona_lora")]:
                run("pairs", "--baseline", generations / f"{baseline}.json",
                    "--candidate", generations / f"{candidate}.json",
                    "--output", OUTPUT_ROOT / "pairs" / f"{track}.jsonl", "--track", track)
    else:
        print("Inspect the sanity responses, then set RUN_FULL=True.")
    '''), SAVE]
    write("04_generate.ipynb", cells)

    cells = shared("05 · Kaggle judging and report evidence", "Create a benchmark-enabled Kaggle notebook at https://www.kaggle.com/benchmarks/tasks/new, then import/paste these cells. Attach saved comparison outputs; attach training outputs to build plots. This notebook never generates model responses.", '''
    PAIRS_ROOT = Path("/kaggle/input/YOUR-GENERATION-OUTPUT/outputs/pairs")
    TRACKS = ["hh", "ultrafeedback", "persona"]
    JUDGE_MODEL = "google/gemini-3.6-flash"  # Select an available Kaggle judge if unavailable.
    JUDGE_RESUME_ROOT = None  # saved outputs/judging containing hh/, ultrafeedback/, persona/
    BUILD_REPORT = True
    TRAINING_RUNS = {  # Attach these output datasets for plots and actual configurations.
        "sft": "/kaggle/input/YOUR-SFT-OUTPUT/outputs/sft",
        "hh": "/kaggle/input/YOUR-HH-OUTPUT/outputs/dpo_hh",
        "ultrafeedback": "/kaggle/input/YOUR-UF-OUTPUT/outputs/dpo_ultrafeedback",
        "persona": "/kaggle/input/YOUR-PERSONA-OUTPUT/outputs/persona",
    }
    ''', judge=True)
    cells += [markdown("## 3. Preflight and judge smoke check\n\nRequests use the Kaggle SDK in this notebook kernel so benchmark context remains available. The first two pairs test access and JSON parsing. Each attempt waits at least five seconds."), code('''
    from alignment.judge import judge_pairs
    from alignment.utils import read_jsonl, write_jsonl
    for track in TRACKS:
        need(PAIRS_ROOT / f"{track}.jsonl", f"{track} pairwise comparison")
    if RUN_SMOKE:
        track = TRACKS[0]
        smoke_pairs = OUTPUT_ROOT / "judge_smoke_pairs.jsonl"
        write_jsonl(smoke_pairs, read_jsonl(PAIRS_ROOT / f"{track}.jsonl")[:2])
        summary = judge_pairs(smoke_pairs, OUTPUT_ROOT / "judge_smoke", judge_model=JUDGE_MODEL)
        print(json.dumps(summary, indent=2))
        assert summary["quality"]["failures"] == 0, "Resolve model access/parsing failures before the full run."
    '''), markdown("## 4. Full resumable judging"), code('''
    if RUN_FULL:
        for track in TRACKS:
            resume = Path(JUDGE_RESUME_ROOT) / track if JUDGE_RESUME_ROOT else None
            summary = judge_pairs(PAIRS_ROOT / f"{track}.jsonl", OUTPUT_ROOT / "judging" / track,
                judge_model=JUDGE_MODEL, batch_size=5, sleep_seconds=5, max_retries=3, resume_from=resume)
            print(json.dumps(summary, indent=2))
    else:
        print("Smoke only. Set RUN_FULL=True once judge access and parsing work.")
    '''), markdown("## 5. Build plots, tables, examples, and a report draft\n\nComplete the draft's interpretation and limitations using the actual results. Export the final PDF and check it is at most eight pages including references."), code('''
    if RUN_FULL and BUILD_REPORT:
        from alignment.report import build_report
        for name, directory in TRAINING_RUNS.items():
            need(Path(directory) / "run.json", f"{name} training run")
        build_report(TRAINING_RUNS, {t: OUTPUT_ROOT / "judging" / t for t in TRACKS},
            OUTPUT_ROOT / "report", {t: PAIRS_ROOT / f"{t}.jsonl" for t in TRACKS})
        print((OUTPUT_ROOT / "report/report_draft.md").read_text())
    '''), SAVE]
    write("05_judge_report.ipynb", cells, gpu=False)


if __name__ == "__main__":
    main()
