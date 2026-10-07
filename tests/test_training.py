import copy
from types import SimpleNamespace
import sys

import pytest
import torch

from alignment import core
from alignment.train import load_state, optimizer_step, run_training, save_state, scheduler_for
from alignment.utils import atomic_json, read_json, read_jsonl, seed_everything
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_accumulated_token_mean_matches_full_batch_including_short_tail():
    tok = ToyTokenizer()
    full, accumulated = ToyCausalLM(tok.vocab_size, .2), ToyCausalLM(tok.vocab_size, .2)
    prompts, responses = ["q", "ab", "c"], ["a", "longer", "z"]
    core.compute_sft_loss(full, tok, prompts, responses, 20).backward()
    batches = [core.build_sft_batch(tok, prompts[i:i + 2], responses[i:i + 2], 20) for i in (0, 2)]
    count = sum(int(core.response_token_count(b)) for b in batches)
    for batch in batches:
        (-core.response_logps_from_batch(accumulated, batch).sum() / count).backward()
    torch.testing.assert_close(full.logit_bias.grad, accumulated.logit_bias.grad)


def test_checkpoint_restores_optimizer_scheduler_scaler_and_rng(tmp_path):
    model = ToyCausalLM(10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = scheduler_for(optimizer, {"scheduler": "cosine", "warmup_ratio": .1}, 10)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    seed_everything(42)
    model.logit_bias.sum().backward()
    optimizer_step(model, optimizer, scaler, scheduler, 1.)
    expected_parameters = copy.deepcopy(model.state_dict())
    progress = {"step": 1, "epoch": 0, "next_group": 1}
    path = tmp_path / "resume.pt"
    save_state(path, model, optimizer, scheduler, scaler, progress, "abc", [{"step": 1}])
    expected_random = torch.rand(4)
    with torch.no_grad():
        model.logit_bias.add_(1)
    state, metrics = load_state(path, model, optimizer, scheduler, scaler, "abc")
    assert state == progress and metrics == [{"step": 1}]
    torch.testing.assert_close(torch.rand(4), expected_random)
    torch.testing.assert_close(model.state_dict()["logit_bias"], expected_parameters["logit_bias"])
    assert scheduler.last_epoch == 1 and optimizer.state
    with pytest.raises(ValueError, match="differ"):
        load_state(path, model, optimizer, scheduler, scaler, "wrong")


class SavedToyTokenizer(ToyTokenizer):
    def __init__(self):
        super().__init__()
        for char in "#\n":
            self.vocab[char] = len(self.vocab)

    def save_pretrained(self, path):
        atomic_json(path / "tokenizer_config.json", {"toy": True})


class SavedToyModel(ToyCausalLM):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.config = SimpleNamespace(use_cache=True, _commit_hash="toy-fixed")

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def save_pretrained(self, path, **kwargs):
        torch.save(self.state_dict(), path / "toy_weights.pt")


def test_training_loop_and_interrupted_resume_match_uninterrupted(tmp_path, monkeypatch):
    tokenizer = SavedToyTokenizer()
    # Patch only loading APIs, so the real loop/math/checkpoint code runs offline.
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: SavedToyModel(tokenizer.vocab_size)))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    data = tmp_path / "data"
    rows = [{"id": str(i), "prompt": "q", "response": "a" * (i % 3 + 1)} for i in range(8)]
    atomic_json(data / "train.json", rows)
    atomic_json(data / "validation.json", rows[:2])
    atomic_json(data / "manifest.json", {"source": "toy"})
    settings = {"policy_device": "cpu", "epochs": 2, "batch_size": 2, "accumulation": 2,
                "max_length": 64, "eval_steps": 1, "learning_rate": .01}
    full = run_training("sft", data, tmp_path / "full", **settings)
    from alignment import train
    original_save = train.save_state
    def interrupt_after_checkpoint(*args, **kwargs):
        original_save(*args, **kwargs)
        if args[5]["step"] == 1:
            raise RuntimeError("simulated disconnect")
    monkeypatch.setattr(train, "save_state", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated"):
        run_training("sft", data, tmp_path / "interrupted", **settings)
    monkeypatch.setattr(train, "save_state", original_save)
    resumed = run_training("sft", data, tmp_path / "resumed", resume=tmp_path / "interrupted/resume.pt", **settings)
    a = torch.load(full / "final/toy_weights.pt", weights_only=True)
    b = torch.load(resumed / "final/toy_weights.pt", weights_only=True)
    torch.testing.assert_close(a["logit_bias"], b["logit_bias"])
    assert read_json(full / "summary.json") == read_json(resumed / "summary.json")
    assert [r["step"] for r in read_jsonl(full / "metrics.jsonl") if r["event"] == "train"] == [1, 2, 3, 4]


def test_dpo_training_loop_tracks_rewards(tmp_path, monkeypatch):
    tokenizer = SavedToyTokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: SavedToyModel(tokenizer.vocab_size))))
    data = tmp_path / "data"
    rows = [{"id": str(i), "prompt": "q", "chosen": "a", "rejected": "b"} for i in range(4)]
    for split in ("train", "validation"):
        atomic_json(data / f"{split}.json", rows)
    atomic_json(data / "manifest.json", {"source": "toy"})
    checkpoint = tmp_path / "sft"
    atomic_json(checkpoint / "alignment_manifest.json", {"kind": "sft", "fingerprint": "sft-toy"})
    output = run_training("dpo", data, tmp_path / "dpo", model=checkpoint, policy_device="cpu", reference_device="cpu",
                          batch_size=1, accumulation=2, max_length=64, max_prompt_length=32, eval_steps=1,
                          learning_rate=.1)
    validations = [r for r in read_jsonl(output / "metrics.jsonl") if r["event"] == "validation"]
    assert validations[0]["reward_accuracy"] == 0
    assert validations[-1]["reward_margin"] > 0
    assert read_json(output / "summary.json")["completed"]
