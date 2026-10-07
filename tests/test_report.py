from alignment.report import build_report
from alignment.utils import atomic_json, read_json, write_jsonl


def test_report_uses_measured_configs_results_and_produces_plots(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    from PIL import Image
    run = tmp_path / "sft"
    config = {"learning_rate": 2e-5, "scheduler": "cosine", "warmup_ratio": .03,
              "batch_size": 2, "accumulation": 16, "max_length": 512, "epochs": 1, "seed": 42}
    atomic_json(run / "run.json", {"kind": "sft", "config": config})
    atomic_json(run / "summary.json", {"best_step": 1})
    write_jsonl(run / "metrics.jsonl", [{"event": "train", "step": 1, "train_loss": 2.},
                                        {"event": "validation", "step": 1, "val_loss": 2.1}])
    judging = tmp_path / "judging"
    quality = {"total": 3, "wins": 2, "ties": 1, "losses": 0, "unknown": 0, "failures": 0,
               "win_rate_without_ties": 1., "tie_adjusted_win_rate": 2.5 / 3}
    atomic_json(judging / "summary.json", {"judge_model": "mock-judge", "quality": quality,
                                           "request_attempts": 1, "retry_attempts": 0})
    output = build_report({"sft": run}, {"hh": judging}, tmp_path / "report")
    assert read_json(output / "configurations.json")["sft"]["config"] == config
    draft = (output / "report_draft.md").read_text()
    assert "{{" not in draft and "83.3%" in draft and "TODO" in draft
    assert "mock-judge" in (output / "results.csv").read_text()
    with Image.open(output / "figures/sft.png") as image:
        assert image.width > 500 and image.height > 300
