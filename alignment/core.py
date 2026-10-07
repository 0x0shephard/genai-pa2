"""Response-only causal LM math, implemented independently of Trainer/TRL.

Prompts passed to the adapter are already formatted. Encode prompt and response
separately to make their token boundary explicit; disable automatic special tokens.
"""
import torch
import torch.nn.functional as F


def encode_example(tokenizer, prompt, response, max_length, max_prompt_length=None):
    if max_length < 2:
        raise ValueError("max_length must be at least 2 for causal scoring")
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if max_prompt_length is not None:
        # Training-only cap; retain the end containing the assistant header.
        prompt_ids = prompt_ids[-max_prompt_length:]
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        response_ids.append(eos)
    ids = (prompt_ids + response_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
    return {"input_ids": ids, "labels": labels}


def pad_examples(tokenizer, examples):
    if not examples:
        raise ValueError("An LM batch must contain at least one row")
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if pad is None:
        pad = 0  # Masked synthetic padding for tokenizers without pad/EOS.
    width = max(len(row["input_ids"]) for row in examples)
    if width == 0:
        raise ValueError("Cannot batch entirely empty sequences")
    ids, masks, labels = [], [], []
    for row in examples:
        n = len(row["input_ids"])
        ids.append(row["input_ids"] + [pad] * (width - n))
        masks.append([1] * n + [0] * (width - n))
        labels.append(row["labels"] + [-100] * (width - n))
    return {k: torch.tensor(v, dtype=torch.long) for k, v in
            (("input_ids", ids), ("attention_mask", masks), ("labels", labels))}


def build_sft_batch(tokenizer, prompts, responses, max_length):
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have equal lengths")
    return pad_examples(tokenizer, [encode_example(tokenizer, p, r, max_length)
                                    for p, r in zip(prompts, responses)])


def model_device(model):
    return next(model.parameters()).device


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def response_logps_from_batch(model, batch):
    batch = move_batch(batch, model_device(model))
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = output.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100) & batch["attention_mask"][:, 1:].bool()
    # Only materialize fp32 vocabulary scores for response positions.
    token_logps = -F.cross_entropy(logits[mask].float(), labels[mask], reduction="none")
    row_ids = torch.arange(labels.shape[0], device=labels.device)[:, None].expand_as(labels)[mask]
    return logits.new_zeros(labels.shape[0], dtype=torch.float32).index_add(0, row_ids, token_logps)


def response_token_count(batch):
    return (batch["labels"][:, 1:].ne(-100) & batch["attention_mask"][:, 1:].bool()).sum()


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return response_logps_from_batch(model, build_sft_batch(tokenizer, prompts, responses, max_length))


def sft_loss_from_batch(model, batch):
    count = response_token_count(batch)
    if count.item() == 0:
        raise ValueError("Batch has no scoreable response tokens after truncation")
    return -response_logps_from_batch(model, batch).sum() / count.to(model_device(model))


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    return sft_loss_from_batch(model, build_sft_batch(tokenizer, prompts, responses, max_length))


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("DPO inputs must have equal lengths")
    return {"chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
            "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length)}


def dpo_terms(pc, pr, rc, rr, beta):
    if beta <= 0:
        raise ValueError("beta must be positive")
    if pc.ndim != 1 or not (pc.shape == pr.shape == rc.shape == rr.shape) or pc.numel() == 0:
        raise ValueError("DPO log probabilities must be nonempty, equally sized row vectors")
    rc, rr = rc.detach().to(pc.device), rr.detach().to(pc.device)
    chosen_reward = beta * (pc - rc)
    rejected_reward = beta * (pr - rr)
    margin = chosen_reward - rejected_reward
    losses = -F.logsigmoid(margin)
    return losses, (margin > 0).float(), margin


def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                              reference_chosen_logps, reference_rejected_logps, beta):
    return dpo_terms(policy_chosen_logps, policy_rejected_logps,
                     reference_chosen_logps, reference_rejected_logps, beta)[0].mean()


def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    pc = response_logps_from_batch(policy_model, batch["chosen"])
    pr = response_logps_from_batch(policy_model, batch["rejected"])
    with torch.no_grad():
        rc = response_logps_from_batch(reference_model, batch["chosen"])
        rr = response_logps_from_batch(reference_model, batch["rejected"])
    return compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
