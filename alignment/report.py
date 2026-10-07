"""Create evidence tables, plots and a report draft from real saved runs."""
import csv
from pathlib import Path

from .utils import atomic_json, read_json, read_jsonl, require_path


def plot_run(run_dir, output, name=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir = require_path(run_dir, "training run")
    metadata = read_json(run_dir / "run.json")
    name = name or metadata["kind"]
    if metadata["kind"] == "persona":
        history = read_json(run_dir / "trainer_history.json")
        records = []
        for row in history:
            record = {"step": row["step"]}
            for original, new in (("loss", "train_loss"), ("eval_loss", "val_loss"),
                                  ("eval_rewards/accuracies", "reward_accuracy"),
                                  ("eval_rewards/margins", "reward_margin")):
                if original in row:
                    record[new] = row[original]
            records.append(record)
    else:
        records = read_jsonl(run_dir / "metrics.jsonl")
    is_dpo = metadata["kind"] != "sft"
    fig, axes = plt.subplots(1, 3 if is_dpo else 1, figsize=(12 if is_dpo else 6, 3.6), squeeze=False)
    for key, label in (("train_loss", "Training"), ("val_loss", "Validation")):
        points = [r for r in records if key in r]
        if points:
            axes[0, 0].plot([r["step"] for r in points], [r[key] for r in points], label=label, linewidth=1.2)
    axes[0, 0].set_ylabel("Response-token mean NLL" if not is_dpo else "DPO loss")
    axes[0, 0].legend()
    if is_dpo:
        for i, key, label in ((1, "reward_accuracy", "Validation reward accuracy"),
                              (2, "reward_margin", "Validation reward margin")):
            points = [r for r in records if key in r]
            if points:
                axes[0, i].plot([r["step"] for r in points], [r[key] for r in points], linewidth=1.2)
            axes[0, i].set_ylabel(label)
    summary = read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
    best_step = summary.get("best_step")
    if best_step is None and summary.get("best_checkpoint"):
        best_step = int(summary["best_checkpoint"].rsplit("-", 1)[-1])
    for ax in axes[0]:
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=.2)
        if best_step is not None:
            ax.axvline(best_step, color="gray", linestyle="--", alpha=.7)
    fig.suptitle(f"{name}; raw metrics, no smoothing; dashed line = selected checkpoint")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def build_report(run_dirs, judge_dirs, output, comparisons=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configurations, summaries = {}, {}
    for name, directory in run_dirs.items():
        directory = require_path(directory, name)
        configurations[name] = read_json(directory / "run.json")
        plot_run(directory, output / "figures" / f"{name}.png", name)
    for name, directory in judge_dirs.items():
        summaries[name] = read_json(require_path(Path(directory) / "summary.json", name))
    atomic_json(output / "configurations.json", configurations)
    atomic_json(output / "evaluation_summary.json", summaries)
    table = []
    for track, summary in summaries.items():
        for dimension in ("quality", "style"):
            if dimension in summary:
                table.append({"track": track, "dimension": dimension, "judge": summary["judge_model"],
                    **{k: summary[dimension][k] for k in ("total", "wins", "ties", "losses", "unknown", "failures",
                                                       "win_rate_without_ties", "tie_adjusted_win_rate")},
                    "request_attempts": summary["request_attempts"], "retry_attempts": summary["retry_attempts"]})
    if table:
        with (output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    configuration_rows = []
    for name, metadata in configurations.items():
        config = metadata["config"]
        configuration_rows.append(f"| {name} | {'LoRA r=' + str(config['rank']) if metadata['kind'] == 'persona' else 'Full'} | "
            f"{config['learning_rate']:g} | {config['scheduler']} / {config['warmup_ratio']:.0%} | "
            f"{config['batch_size']} × {config['accumulation']} | {config['max_length']} | {config['epochs']} | "
            f"{config.get('beta', '—')} | {config['seed']} |")
    results_rows = []
    def percent(value):
        return "N/A" if value is None else f"{value:.1%}"
    for row in table:
        results_rows.append(f"| {row['track']} / {row['dimension']} | {row['total']} | {row['wins']} | {row['ties']} | "
            f"{row['losses']} | {row['unknown']} | {row['failures']} | {percent(row['win_rate_without_ties'])} | "
            f"{percent(row['tie_adjusted_win_rate'])} |")
    examples = []
    for name, pairs_path in (comparisons or {}).items():
        lookup = {p["id"]: p for p in read_jsonl(pairs_path)}
        judgments = read_jsonl(Path(judge_dirs[name]) / "judgments.jsonl")
        buckets = {"improvement": [], "regression": [], "little_change": []}
        for row in judgments:
            pair = lookup[row["id"]]
            winner = row["winner"]
            if winner == "UNKNOWN":
                continue
            mapping = pair["hidden_metadata"]
            candidate_side = "a" if mapping["model_a"] == mapping["candidate"] else "b"
            outcome = "little_change" if winner == "TIE" else (
                "improvement" if winner.lower() == candidate_side else "regression")
            if len(buckets[outcome]) < 2:
                buckets[outcome].append({"track": name, "outcome": outcome, "id": row["id"],
                    "instruction": pair["instruction"], "candidate": pair[f"response_{candidate_side}"],
                    "baseline": pair["response_b" if candidate_side == "a" else "response_a"], "judge_reason": row["reason"]})
        examples.extend(example for values in buckets.values() for example in values)
    atomic_json(output / "qualitative_examples.json", examples)
    template = (Path(__file__).resolve().parents[1] / "docs/report_template.md").read_text()
    template = template.replace("{{CONFIGURATION_TABLE}}", "\n".join(configuration_rows) or "No completed training runs supplied.")
    template = template.replace("{{RESULTS_TABLE}}", "\n".join(results_rows) or "No completed judgments supplied.")
    (output / "report_draft.md").write_text(template)
    print(f"Report evidence saved to {output}. Complete interpretation and export a PDF of at most 8 pages.")
    return output
