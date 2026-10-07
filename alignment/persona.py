"""The only training track allowed to use TRL DPOTrainer and PEFT."""
import os
from pathlib import Path
import shutil

from .utils import atomic_json, environment, read_json, require_path, seed_everything, stable_hash


def run_persona(data_dir, output, max_steps=None, resume=None, seed=42):
    # Trainer would otherwise use DataParallel and silently double the batch size.
    # Call this stage in its own process, as the supplied notebook does.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle T4 GPUs for persona training")
    seed_everything(seed)
    output = Path(output)
    if (output / "run.json").exists() and resume is None:
        raise ValueError("Persona run already exists; use --resume or a new output directory")
    data_dir = require_path(data_dir, "prepared persona data")
    rows = {split: read_json(data_dir / f"{split}.json") for split in ("train", "validation")}
    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
    previous = None
    if resume:
        resume = require_path(resume, "TRL checkpoint")
        previous = read_json(require_path(resume.parent.parent / "run.json", "persona resume manifest"))
    revision = previous["config"]["model_revision"] if previous else None
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ValueError("SmolLM2 tokenizer must provide a chat template")
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.float32,
                                               attn_implementation="sdpa").to("cuda:0")
    model.config.use_cache = False
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    if set(targets) - present:
        raise ValueError(f"Missing LoRA targets: {set(targets) - present}")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=.05, target_modules=targets,
                      bias="none", task_type="CAUSAL_LM")
    def conversational(row):
        return {"prompt": [{"role": "user", "content": row["prompt"]}],
                "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}]}
    datasets = {k: Dataset.from_list([conversational(r) for r in v]) for k, v in rows.items()}
    config = {"model": model_id, "model_revision": getattr(model.config, "_commit_hash", None),
              "rank": 16, "alpha": 32, "dropout": .05, "targets": targets,
              "batch_size": 2, "accumulation": 16, "effective_batch_size": 32,
              "learning_rate": 5e-5, "epochs": 3, "max_length": 512, "max_prompt_length": 256,
              "beta": .1, "scheduler": "cosine", "warmup_ratio": .03, "weight_decay": 0.,
              "optimizer": "AdamW", "adam_betas": [.9, .999], "adam_epsilon": 1e-8,
              "clip": 1., "seed": seed, "max_steps": max_steps,
              "precision": "fp32 weights and fp16 autocast", "reference": "base model with adapter disabled",
              "chat_template": tokenizer.chat_template, "chat_template_hash": stable_hash(tokenizer.chat_template)}
    metadata = {"kind": "persona", "config": config, "environment": environment(),
                "actual_examples": {k: len(v) for k, v in rows.items()},
                "data_manifest": read_json(data_dir / "manifest.json")}
    if previous and (previous["config"] != config or previous["data_manifest"] != metadata["data_manifest"]):
        raise ValueError("Persona resume configuration/data differ from the saved run")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "run.json", metadata)
    eval_steps = 1 if max_steps is not None else 10
    args = DPOConfig(output_dir=str(output / "trainer"), num_train_epochs=3,
        per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=16,
        learning_rate=5e-5, lr_scheduler_type="cosine", warmup_ratio=.03, weight_decay=0.,
        max_grad_norm=1., beta=.1, max_length=512, max_prompt_length=256,
        disable_dropout=False,  # Retain the declared 0.05 LoRA dropout (base dropout is zero).
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps,
        logging_steps=1, save_total_limit=2, load_best_model_at_end=False,
        metric_for_best_model="eval_rewards/accuracies", greater_is_better=True,
        report_to="none", seed=seed, data_seed=seed, max_steps=max_steps or -1,
        optim="adamw_torch", remove_unused_columns=False)

    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, **kwargs):
            atomic_json(output / "trainer_history.json", state.log_history)

    # With PEFT and ref_model=None, TRL evaluates the unchanged base by disabling
    # the adapter. A reward model or an additional full model is unnecessary.
    trainer = DPOTrainer(model=model, ref_model=None, args=args, processing_class=tokenizer,
                         train_dataset=datasets["train"], eval_dataset=datasets["validation"],
                         peft_config=lora, callbacks=[MetricsCallback()])
    metadata["trainable_parameters"] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    atomic_json(output / "run.json", metadata)
    trainer.train(resume_from_checkpoint=str(require_path(resume, "TRL checkpoint")) if resume else None)
    final_metrics = trainer.evaluate()
    trainer.save_model(str(output / "final"))
    tokenizer.save_pretrained(output / "final")
    atomic_json(output / "final/alignment_manifest.json", metadata)
    checkpoint = trainer.state.best_model_checkpoint
    best_metric = trainer.state.best_metric
    final_accuracy = final_metrics["eval_rewards/accuracies"]
    final_is_best = best_metric is None or final_accuracy > best_metric
    if final_is_best:
        shutil.copytree(output / "final", output / "best", dirs_exist_ok=True)
        checkpoint = str(output / "final")
        best_metric = final_accuracy
        best_step = trainer.state.global_step
    elif checkpoint:
        # On a fresh Kaggle session, state may still refer to /kaggle/working from
        # the earlier session. Resolve the checkpoint within attached outputs.
        if not Path(checkpoint).exists() and resume:
            checkpoint = str(require_path(resume.parent / Path(checkpoint).name, "previous TRL best checkpoint"))
        shutil.copytree(checkpoint, output / "best", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt", "rng_state*.pth", "scaler.pt"))
        tokenizer.save_pretrained(output / "best")
        best_step = int(Path(checkpoint).name.rsplit("-", 1)[-1])
    else:
        raise RuntimeError("TRL did not select a best checkpoint; inspect validation logs")
    atomic_json(output / "best/alignment_manifest.json", {**metadata, "selected_checkpoint": checkpoint})
    atomic_json(output / "trainer_history.json", trainer.state.log_history)
    atomic_json(output / "summary.json", {"step": trainer.state.global_step,
        "best_checkpoint": checkpoint, "best_step": best_step, "best_metric": best_metric,
        "final_validation": final_metrics, "completed": True})
    return output
