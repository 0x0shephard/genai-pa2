from pathlib import Path
import random

from .utils import (PROMPT_TEMPLATE, atomic_json, environment, format_prompt, read_json,
                    require_path, seed_everything, stable_hash, write_jsonl)


def load_generation_model(checkpoint, persona=False, device="cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = {"torch_dtype": torch.float16 if device.startswith("cuda") else torch.float32,
              "attn_implementation": "sdpa"}
    path = Path(checkpoint)
    if path.exists():
        manifest = read_json(path / "alignment_manifest.json")
        if (path / "adapter_config.json").exists():
            from peft import PeftModel
            adapter = read_json(path / "adapter_config.json")
            base = AutoModelForCausalLM.from_pretrained(adapter["base_model_name_or_path"],
                revision=manifest["config"]["model_revision"], **kwargs)
            model = PeftModel.from_pretrained(base, path).merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(path)
        identity = {"checkpoint": str(path), "manifest": manifest}
    else:
        if checkpoint not in ("Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-360M-Instruct"):
            require_path(checkpoint, "generation checkpoint")
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
        revision = getattr(model.config, "_commit_hash", None)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        identity = {"model": checkpoint, "revision": revision}
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device).eval()
    model.config.use_cache = True
    if persona and not tokenizer.chat_template:
        raise ValueError("Persona generation requires the model's chat template")
    return model, tokenizer, identity


def generate_outputs(checkpoint, prompts_path, output_path, generator, persona=False,
                     max_new_tokens=256, repetition_penalty=1., no_repeat_ngram_size=0,
                     seed=42, limit=None, resume_from=None, device="cuda:0"):
    import torch
    seed_everything(seed)
    prompts = read_json(require_path(prompts_path, "evaluation prompts"))
    if limit:
        prompts = prompts[:limit]
    ids = [r["id"] for r in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation prompt IDs must be unique")
    output_path = Path(output_path)
    model, tokenizer, identity = load_generation_model(checkpoint, persona, device)
    settings = {"do_sample": False, "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty, "no_repeat_ngram_size": no_repeat_ngram_size,
                "length_penalty": 1., "temperature": None, "top_p": None, "seed": seed,
                "prompt_template": tokenizer.chat_template if persona else PROMPT_TEMPLATE,
                "max_prompt_tokens": 256}
    # The same 256-token prompt cap is applied to every model in a comparison.
    manifest = {"generator": generator, "model_identity": identity, "decoding": settings,
                "prompt_hash": stable_hash(prompts), "count": len(prompts), "environment": environment()}
    fingerprint = stable_hash({k: v for k, v in manifest.items() if k != "environment"})
    manifest["fingerprint"] = fingerprint
    source = output_path if output_path.exists() else (Path(resume_from) if resume_from else None)
    previous = read_json(require_path(source)) if source else []
    if source:
        old = read_json(source.with_suffix(".manifest.json"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("Generation resume differs in checkpoint, prompts, or decoding")
    results = {r["id"]: r for r in previous}
    if len(results) != len(previous) or set(results) - set(ids):
        raise ValueError("Resume generations contain duplicate or unknown IDs")
    atomic_json(output_path.with_suffix(".manifest.json"), manifest)
    with torch.inference_mode():
        for row in prompts:
            if row["id"] in results:
                continue
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["instruction"]}],
                        tokenize=False, add_generation_prompt=True) if persona else format_prompt(row["instruction"])
            tokens = tokenizer.encode(prompt, add_special_tokens=False)[-256:]
            input_ids = torch.tensor([tokens], device=device)
            generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                do_sample=False, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            output = tokenizer.decode(generated[0, len(tokens):], skip_special_tokens=True).strip()
            results[row["id"]] = {**row, "output": output, "generator": generator,
                                   "generation_fingerprint": fingerprint}
            atomic_json(output_path, [results[r["id"]] for r in prompts if r["id"] in results])
            print(f"{generator}: {len(results)}/{len(prompts)}", flush=True)
    return output_path


def build_pairs(first_path, second_path, output, track, seed=42):
    first, second = read_json(first_path), read_json(second_path)
    first_manifest = read_json(Path(first_path).with_suffix(".manifest.json"))
    second_manifest = read_json(Path(second_path).with_suffix(".manifest.json"))
    if first_manifest["prompt_hash"] != second_manifest["prompt_hash"]:
        raise ValueError("Models were generated on different prompt sets")
    if first_manifest["decoding"] != second_manifest["decoding"]:
        raise ValueError("Models must use identical decoding and prompt formatting")
    if first_manifest["generator"] == second_manifest["generator"]:
        raise ValueError("Comparison requires different model identities")
    right = {r["id"]: r for r in second}
    if len(right) != len(second) or len({r["id"] for r in first}) != len(first):
        raise ValueError("Duplicate generation IDs")
    if {r["id"] for r in first} != set(right):
        raise ValueError("Models must have outputs for identical prompt IDs")
    randomizer = random.Random(seed)
    pairs = []
    for left in first:
        other = right[left["id"]]
        if left["instruction"] != other["instruction"]:
            raise ValueError("Prompt content differs")
        a, b = (left, other) if randomizer.random() < .5 else (other, left)
        pairs.append({"id": left["id"], "track": track, "instruction": left["instruction"],
                      "response_a": a["output"], "response_b": b["output"],
                      "category": left.get("category"),
                      "hidden_metadata": {"model_a": a["generator"], "model_b": b["generator"],
                          "baseline": first_manifest["generator"], "candidate": second_manifest["generator"]}})
    write_jsonl(output, pairs)
    atomic_json(Path(output).with_suffix(".manifest.json"), {"seed": seed, "track": track,
        "hash": stable_hash(pairs), "count": len(pairs), "first": first_manifest, "second": second_manifest})
    return Path(output)
