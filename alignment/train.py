"""Ordinary PyTorch SFT/DPO loops; no Hugging Face Trainer or TRL."""
from contextlib import nullcontext
from pathlib import Path
import math
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from . import core
from .utils import (PROMPT_TEMPLATE, atomic_json, environment, format_prompt, read_json,
                    read_jsonl, require_path, restore_rng, save_rng, seed_everything,
                    stable_hash, write_jsonl)


DEFAULTS = {
    "sft": {"model": "Qwen/Qwen2.5-0.5B", "batch_size": 2, "accumulation": 16,
            "learning_rate": 2e-5, "weight_decay": .1, "warmup_ratio": .03, "scheduler": "cosine"},
    "dpo": {"batch_size": 1, "accumulation": 32, "learning_rate": 1e-6,
            "weight_decay": 0., "warmup_ratio": 0., "scheduler": "constant", "beta": .1,
            "max_prompt_length": 256},
}


def autocast(device):
    device = torch.device(device)
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def prepare_examples(rows, tokenizer, config):
    encoded, dropped = [], []
    for row in rows:
        prompt = format_prompt(row["prompt"])
        sides = ("response",) if config["kind"] == "sft" else ("chosen", "rejected")
        record = {side: core.encode_example(tokenizer, prompt, row[side], config["max_length"],
                                           config.get("max_prompt_length")) for side in sides}
        if any(not any(x != -100 for x in value["labels"][1:]) for value in record.values()):
            dropped.append(row["id"])
        else:
            encoded.append(record)
    if not encoded:
        raise ValueError("All examples lack response tokens after truncation")
    return encoded, dropped


def collate(rows, tokenizer, kind):
    if kind == "sft":
        return core.pad_examples(tokenizer, [r["response"] for r in rows])
    return {side: core.pad_examples(tokenizer, [r[side] for r in rows]) for side in ("chosen", "rejected")}


def dpo_forward(policy, reference, batch, beta):
    with autocast(core.model_device(policy)):
        pc = core.response_logps_from_batch(policy, batch["chosen"])
        pr = core.response_logps_from_batch(policy, batch["rejected"])
    with torch.no_grad(), autocast(core.model_device(reference)):
        rc = core.response_logps_from_batch(reference, batch["chosen"])
        rr = core.response_logps_from_batch(reference, batch["rejected"])
    return core.dpo_terms(pc, pr, rc, rr, beta)


def validate(policy, reference, examples, tokenizer, config):
    was_training = policy.training
    policy.eval()
    totals, denominator = [0., 0., 0.], 0
    with torch.no_grad():
        for start in range(0, len(examples), config["batch_size"]):
            rows = examples[start:start + config["batch_size"]]
            batch = collate(rows, tokenizer, config["kind"])
            if config["kind"] == "sft":
                with autocast(core.model_device(policy)):
                    totals[0] += float(-core.response_logps_from_batch(policy, batch).sum())
                denominator += int(core.response_token_count(batch))
            else:
                losses, accuracy, margin = dpo_forward(policy, reference, batch, config["beta"])
                for i, values in enumerate((losses, accuracy, margin)):
                    totals[i] += float(values.sum())
                denominator += len(rows)
    policy.train(was_training)
    result = {"val_loss": totals[0] / denominator}
    if config["kind"] == "dpo":
        result.update(reward_accuracy=totals[1] / denominator, reward_margin=totals[2] / denominator)
    return result


def optimizer_step(model, optimizer, scaler, scheduler, clip):
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= previous_scale
    if updated:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm), updated


def scheduler_for(optimizer, config, total_steps):
    warmup = math.ceil(total_steps * config["warmup_ratio"])
    def factor(step):
        if config["scheduler"] == "constant":
            return 1.
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return .5 * (1 + math.cos(math.pi * min(progress, 1.)))
    return LambdaLR(optimizer, factor)


def save_state(path, policy, optimizer, scheduler, scaler, progress, fingerprint, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "progress": progress, "rng": save_rng(), "fingerprint": fingerprint,
                "metrics": metrics}, temporary)
    temporary.replace(path)


def load_state(path, policy, optimizer, scheduler, scaler, fingerprint):
    # Only load your own training artifacts: optimizer/RNG checkpoints use pickle.
    state = torch.load(require_path(path, "resume state"), map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint:
        raise ValueError("Resume configuration/data differ from the saved run")
    policy.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state["progress"], state["metrics"]


def save_model(path, policy, tokenizer, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    atomic_json(path / "alignment_manifest.json", metadata)


def run_training(kind, data_dir, output, model=None, resume=None, **overrides):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config = {"kind": kind, "optimizer": "AdamW", "adam_betas": [.9, .999], "adam_epsilon": 1e-8,
              "epochs": 1, "max_length": 512, "clip": 1., "seed": 42,
              "eval_steps": 100, "max_steps": None, "gradient_checkpointing": True,
              "policy_device": "cuda:0", "reference_device": "cuda:1", **DEFAULTS[kind]}
    config.update({k: v for k, v in overrides.items() if v is not None})
    if model:
        config["model"] = str(model)
    if "model" not in config:
        raise ValueError("DPO needs the SFT best checkpoint path")
    if kind == "dpo":
        require_path(config["model"], "SFT best checkpoint")
        source_manifest = read_json(Path(config["model"]) / "alignment_manifest.json")
        if source_manifest["kind"] != "sft":
            raise ValueError("Each DPO track must start from the SFT checkpoint")
    else:
        source_manifest = None
    for field in ("epochs", "max_length", "batch_size", "accumulation", "eval_steps"):
        if config[field] <= 0:
            raise ValueError(f"{field} must be positive")
    if config.get("max_steps") is not None and config["max_steps"] <= 0:
        raise ValueError("max_steps must be positive")
    if torch.device(config["policy_device"]).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle GPU T4 x2 before training")
    if kind == "dpo" and config["reference_device"] == "cuda:1" and torch.cuda.device_count() < 2:
        raise RuntimeError("From-scratch DPO requires both T4 GPUs")
    seed_everything(config["seed"])
    output = Path(output)
    if (output / "run.json").exists() and resume is None:
        raise ValueError("Run directory already exists; use --resume or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    data_dir = require_path(data_dir, "prepared data directory")
    rows = {split: read_json(data_dir / f"{split}.json") for split in ("train", "validation")}
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    examples, dropped = {}, {}
    for split in rows:
        examples[split], dropped[split] = prepare_examples(rows[split], tokenizer, config)
    # Pre-tokenization happens once, not in every epoch.
    policy = AutoModelForCausalLM.from_pretrained(config["model"], torch_dtype=torch.float32,
                                                 attn_implementation="sdpa").to(config["policy_device"])
    policy.config.use_cache = False
    if config["gradient_checkpointing"]:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = None
    if kind == "dpo":
        reference = AutoModelForCausalLM.from_pretrained(config["model"],
            torch_dtype=torch.float16 if torch.device(config["reference_device"]).type == "cuda" else torch.float32,
            attn_implementation="sdpa").to(config["reference_device"])
        reference.requires_grad_(False).eval()
        reference.config.use_cache = False
        for module in policy.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
    group_size = config["batch_size"] * config["accumulation"]
    groups_per_epoch = math.ceil(len(examples["train"]) / group_size)
    total_steps = groups_per_epoch * config["epochs"]
    if config["max_steps"]:
        total_steps = min(total_steps, config["max_steps"])
    optimizer = AdamW(policy.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
                      betas=tuple(config["adam_betas"]), eps=config["adam_epsilon"])
    scheduler = scheduler_for(optimizer, config, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.device(config["policy_device"]).type == "cuda")
    identity_config = {k: v for k, v in config.items() if k not in ("model", "policy_device", "reference_device")}
    model_identity = source_manifest if source_manifest else {"model": config["model"],
                                                            "revision": getattr(policy.config, "_commit_hash", None)}
    fingerprint = stable_hash({"config": identity_config, "data": rows, "initial_model": model_identity})
    metadata = {"kind": kind, "config": config, "fingerprint": fingerprint, "environment": environment(),
                "prompt_template": PROMPT_TEMPLATE, "sft_reduction": "global response-token mean",
                "initial_checkpoint": model_identity, "data_manifest": read_json(data_dir / "manifest.json"),
                "actual_examples": {k: len(v) for k, v in examples.items()}, "dropped_no_response_ids": dropped,
                "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                "precision": "fp32 trainable weights; fp16 CUDA autocast; fp16 frozen reference",
                "effective_batch_size": group_size, "optimizer_steps_planned": total_steps}
    progress = {"epoch": 0, "next_group": 0, "step": 0, "best": None, "best_step": None}
    metrics = []
    if resume:
        progress, metrics = load_state(resume, policy, optimizer, scheduler, scaler, fingerprint)
        # Carry best checkpoint forward when resuming into a fresh Kaggle working directory.
        previous_best = Path(resume).parent / "best"
        if progress["best"] is not None and not (output / "best").exists():
            import shutil
            shutil.copytree(require_path(previous_best, "previous best checkpoint"), output / "best")
    atomic_json(output / "run.json", metadata)
    atomic_json(output / "tokenization.json", {"dropped": dropped})
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()

    def evaluate_and_save():
        results = validate(policy, reference, examples["validation"], tokenizer, config)
        metrics.append({"event": "validation", "step": progress["step"], **results})
        key = [results["val_loss"]] if kind == "sft" else [-results["reward_accuracy"], results["val_loss"]]
        if progress["best"] is None or key < progress["best"]:
            progress["best"], progress["best_step"] = key, progress["step"]
            save_model(output / "best", policy, tokenizer, {**metadata, "selected_step": progress["step"], "validation": results})
        write_jsonl(output / "metrics.jsonl", metrics)
        save_state(output / "resume.pt", policy, optimizer, scheduler, scaler, progress, fingerprint, metrics)
        print({"step": progress["step"], **results}, flush=True)

    if not metrics:
        evaluate_and_save()
    policy.train()
    for epoch in range(progress["epoch"], config["epochs"]):
        order = torch.randperm(len(examples["train"]), generator=torch.Generator().manual_seed(config["seed"] + epoch)).tolist()
        first_group = progress["next_group"] if epoch == progress["epoch"] else 0
        for group_index in range(first_group, groups_per_epoch):
            if progress["step"] >= total_steps:
                break
            selected = [examples["train"][i] for i in order[group_index * group_size:(group_index + 1) * group_size]]
            batches = [collate(selected[i:i + config["batch_size"]], tokenizer, kind)
                       for i in range(0, len(selected), config["batch_size"])]
            denominator = sum(int(core.response_token_count(b)) for b in batches) if kind == "sft" else len(selected)
            train_loss, train_accuracy, train_margin = 0., 0., 0.
            for batch in batches:
                if kind == "sft":
                    with autocast(config["policy_device"]):
                        numerator = -core.response_logps_from_batch(policy, batch).sum()
                    loss = numerator / denominator
                    train_loss += float(numerator.detach()) / denominator
                else:
                    losses, accuracy, margin = dpo_forward(policy, reference, batch, config["beta"])
                    loss = losses.sum() / denominator
                    train_loss += float(losses.detach().sum()) / denominator
                    train_accuracy += float(accuracy.sum()) / denominator
                    train_margin += float(margin.detach().sum()) / denominator
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss; inspect data and precision settings")
                scaler.scale(loss).backward()
            norm, updated = optimizer_step(policy, optimizer, scaler, scheduler, config["clip"])
            progress.update(epoch=epoch, next_group=group_index + 1, step=progress["step"] + int(updated))
            row = {"event": "train", "step": progress["step"], "epoch": epoch,
                   "train_loss": train_loss, "learning_rate": optimizer.param_groups[0]["lr"],
                   "grad_norm": norm if math.isfinite(norm) else None, "optimizer_updated": updated,
                   "elapsed_seconds": time.monotonic() - started}
            if kind == "dpo":
                row.update(train_reward_accuracy=train_accuracy, train_reward_margin=train_margin)
            metrics.append(row)
            write_jsonl(output / "metrics.jsonl", metrics)
            if updated and progress["step"] % config["eval_steps"] == 0:
                evaluate_and_save()
        # If stopped at max_steps, retain the exact next-group position for auditing.
        if progress["next_group"] >= groups_per_epoch:
            progress.update(epoch=epoch + 1, next_group=0)
        evaluate_and_save()
        if progress["step"] >= total_steps:
            break
    save_model(output / "final", policy, tokenizer, {**metadata, "selected_step": progress["step"]})
    atomic_json(output / "summary.json", {"step": progress["step"], "best_step": progress["best_step"],
                                          "best_selection_key": progress["best"], "completed": True})
    return output
