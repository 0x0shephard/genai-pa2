"""Normalize, filter and persist fixed dataset subsets on Kaggle."""
from collections import Counter
from pathlib import Path
import random
import re

from .utils import ROOT, atomic_json, read_json, stable_hash

SOURCES = {
    "sft": ("HuggingFaceH4/ultrachat_200k", "train_sft", "test_sft"),
    "hh": ("Anthropic/hh-rlhf", "train", "test"),
    "ultrafeedback": ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs", "test_prefs"),
}


def normalize_sft(row):
    messages = row.get("messages", [])
    if len(messages) < 2 or [m["role"] for m in messages[:2]] != ["user", "assistant"]:
        return None
    return {"prompt": messages[0]["content"].strip(), "response": messages[1]["content"].strip()}


def normalize_hh(row):
    def split(text):
        if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
            return None
        match = re.fullmatch(r"\s*Human: (.*?)\n\nAssistant: ?(.*)", text, re.S)
        return (match[1].strip(), match[2].strip()) if match else None
    chosen, rejected = split(row["chosen"]), split(row["rejected"])
    if chosen is None or rejected is None or chosen[0] != rejected[0]:
        return None
    return {"prompt": chosen[0], "chosen": chosen[1], "rejected": rejected[1]}


def normalize_uf(row):
    chosen, rejected = row.get("chosen", []), row.get("rejected", [])
    if len(chosen) != 2 or len(rejected) != 2:
        return None
    if any([m["role"] for m in side] != ["user", "assistant"] for side in (chosen, rejected)):
        return None
    if chosen[0]["content"].strip() != rejected[0]["content"].strip():
        return None
    return {"prompt": chosen[0]["content"].strip(), "chosen": chosen[1]["content"].strip(),
            "rejected": rejected[1]["content"].strip()}


def select_rows(rows, normalizer, count, seed, excluded=()):
    counters = Counter()
    valid, seen = [], set(excluded)
    for source_index, row in enumerate(rows):
        counters["source_rows"] += 1
        normalized = normalizer(row)
        if normalized is None or not all(isinstance(v, str) and v.strip() for v in normalized.values()):
            counters["invalid_or_multiturn"] += 1
            continue
        if "chosen" in normalized and normalized["chosen"] == normalized["rejected"]:
            counters["identical_responses"] += 1
            continue
        if normalized["prompt"] in seen:
            counters["duplicate_or_overlapping_prompt"] += 1
            continue
        seen.add(normalized["prompt"])
        normalized["id"] = stable_hash(normalized)[:20]
        normalized["source_index"] = source_index
        valid.append(normalized)
    if len(valid) < count:
        raise ValueError(f"Requested {count} examples, only {len(valid)} valid after filtering: {dict(counters)}")
    random.Random(seed).shuffle(valid)
    counters["valid_rows"] = len(valid)
    counters["selected_rows"] = count
    return valid[:count], dict(counters)


def prepare_track(track, output, train_size=None, val_size=1000, seed=42):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    train_size = train_size or (30000 if track == "sft" else 15000)
    destination = Path(output) / track
    request = {"track": track, "train_size": train_size, "val_size": val_size, "seed": seed}
    if (destination / "manifest.json").exists():
        manifest = read_json(destination / "manifest.json")
        if manifest["request"] != request:
            raise ValueError("Prepared subset settings differ. Choose a new output directory.")
        for split in ("train", "validation"):
            if stable_hash(read_json(destination / f"{split}.json")) != manifest["hashes"][split]:
                raise ValueError("Prepared data hash mismatch")
        return destination
    repo, training_split, validation_split = SOURCES[track]
    revision = HfApi().dataset_info(repo).sha
    normalizer = {"sft": normalize_sft, "hh": normalize_hh, "ultrafeedback": normalize_uf}[track]
    train, train_counts = select_rows(load_dataset(repo, split=training_split, revision=revision),
                                     normalizer, train_size, seed)
    validation, val_counts = select_rows(load_dataset(repo, split=validation_split, revision=revision),
                                        normalizer, val_size, seed, [r["prompt"] for r in train])
    atomic_json(destination / "train.json", train)
    atomic_json(destination / "validation.json", validation)
    atomic_json(destination / "manifest.json", {"request": request, "source": repo, "revision": revision,
        "source_splits": [training_split, validation_split], "filtering": {"train": train_counts, "validation": val_counts},
        "hashes": {"train": stable_hash(train), "validation": stable_hash(validation)}})
    return destination


def prepare_alpaca(output, count=300, seed=42):
    from huggingface_hub import HfApi, hf_hub_download
    destination = Path(output) / f"alpaca_eval_{count}_seed{seed}.json"
    if destination.exists():
        rows = read_json(destination)
        manifest = read_json(destination.with_suffix(".manifest.json"))
        if len(rows) != count or stable_hash(rows) != manifest["hash"]:
            raise ValueError("AlpacaEval subset hash/count mismatch")
        return destination
    repo = "tatsu-lab/alpaca_eval"
    revision = HfApi().dataset_info(repo).sha
    source_path = hf_hub_download(repo, "alpaca_eval.json", repo_type="dataset", revision=revision)
    source = read_json(source_path)
    indices = random.Random(seed).sample(range(len(source)), count)
    rows = [{"id": f"alpaca-{i}", "instruction": source[i]["instruction"], "dataset": "alpaca_eval"} for i in indices]
    atomic_json(destination, rows)
    atomic_json(destination.with_suffix(".manifest.json"), {"source": repo, "revision": revision,
        "seed": seed, "count": count, "indices": indices, "hash": stable_hash(rows)})
    return destination


def prepare_persona(output, seed=42):
    source = read_json(ROOT / "data/persona_lora/ramsay_style_preferences.json")
    rows, counts = select_rows(source, lambda r: {k: r[k].strip() for k in ("prompt", "chosen", "rejected")},
                               len({r["prompt"].strip() for r in source}), seed)
    val_count = max(1, round(len(rows) * .1))
    validation, train = rows[:val_count], rows[val_count:]
    destination = Path(output) / "persona"
    atomic_json(destination / "train.json", train)
    atomic_json(destination / "validation.json", validation)
    atomic_json(destination / "manifest.json", {"seed": seed, "source": "data/persona_lora/ramsay_style_preferences.json",
        "filtering": counts, "sizes": {"train": len(train), "validation": len(validation)},
        "hashes": {"train": stable_hash(train), "validation": stable_hash(validation)}})
    return destination
