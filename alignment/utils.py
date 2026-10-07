import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    # Interrupted writes are prevented by atomic write_jsonl; corrupt files fail loudly.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment():
    packages = {}
    for name in ("torch", "transformers", "datasets", "trl", "peft", "accelerate", "kaggle-benchmarks"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"python": platform.python_version(), "packages": packages,
            "git_commit": result.stdout.strip() if result.returncode == 0 else "uncommitted",
            "git_ref_requested": os.environ.get("ALIGNMENT_GIT_REF")}


def require_path(path, label="input"):
    path = Path(path)
    if not path.exists():
        inputs = Path("/kaggle/input")
        available = [str(p) for p in inputs.glob("*/*")] if inputs.exists() else []
        raise FileNotFoundError(f"Missing {label}: {path}. Attach saved Kaggle outputs and set the path. Available: {available}")
    return path


def save_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
