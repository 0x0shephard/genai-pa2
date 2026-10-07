import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from alignment import core
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class PositionalLM(nn.Module):
    """Position-dependent logits expose a missing causal shift."""
    def __init__(self, vocab):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.2))
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :, None]
        vocabulary = torch.arange(self.vocab, device=input_ids.device)[None, None, :]
        logits = torch.cos(positions - vocabulary) * self.scale
        return SimpleNamespace(logits=logits.expand(input_ids.shape[0], -1, -1))


def test_shifted_response_scoring_matches_manual_calculation():
    tokenizer = ToyTokenizer()
    model = PositionalLM(tokenizer.vocab_size)
    batch = core.build_sft_batch(tokenizer, ["abc", "q"], ["xy", "z"], 20)
    logits = model(batch["input_ids"]).logits.log_softmax(-1)
    expected = []
    for i in range(2):
        expected.append(sum(logits[i, j - 1, batch["labels"][i, j]]
                            for j in range(1, batch["labels"].shape[1]) if batch["labels"][i, j] != -100))
    torch.testing.assert_close(core.response_logps_from_batch(model, batch), torch.stack(expected))


def test_eos_added_before_truncation_and_prompt_only_rows_score_zero():
    tok = ToyTokenizer()
    batch = core.build_sft_batch(tok, ["abc", "abcdef"], ["xyz", "z"], 5)
    assert batch["labels"][0].tolist() == [-100] * 3 + tok.encode("xy")
    assert batch["labels"][1].eq(-100).all()
    logps = core.response_logps_from_batch(ToyCausalLM(tok.vocab_size), batch)
    assert logps[1] == 0
    with pytest.raises(ValueError, match="no scoreable"):
        core.compute_sft_loss(ToyCausalLM(tok.vocab_size), tok, ["abcdef"], ["z"], 5)


def test_padding_equal_to_eos_does_not_mask_real_eos():
    tok = ToyTokenizer()
    tok.pad_token_id = tok.eos_token_id
    batch = core.build_sft_batch(tok, ["a", "long"], ["b", "ccc"], 20)
    assert batch["labels"][0, 2] == tok.eos_token_id
    assert batch["attention_mask"][0, 2] == 1
    assert batch["labels"][0, 3:].eq(-100).all()
    model = ToyCausalLM(tok.vocab_size)
    torch.testing.assert_close(core.response_logps_from_batch(model, batch)[0],
                               torch.tensor(-2 * math.log(tok.vocab_size)))


def test_no_special_tokens_or_missing_pad_and_mixed_order():
    tok = ToyTokenizer()
    tok.eos_token_id = None
    tok.pad_token_id = None
    batch = core.build_sft_batch(tok, ["abc", "q"], ["d", "ef"], 20)
    assert batch["input_ids"][0].tolist() == tok.encode("abcd")
    assert batch["labels"][1].tolist() == [-100] + tok.encode("ef") + [-100]
    logps = core.response_logps_from_batch(ToyCausalLM(tok.vocab_size), batch)
    torch.testing.assert_close(logps, torch.tensor([-math.log(tok.vocab_size), -2 * math.log(tok.vocab_size)]))


def test_dpo_values_and_reference_tensors_are_detached():
    pc = torch.tensor([-2., -5.], requires_grad=True)
    pr = torch.tensor([-4., -3.], requires_grad=True)
    rc = torch.tensor([-3., -4.], requires_grad=True)
    rr = torch.tensor([-4., -4.], requires_grad=True)
    losses, accuracy, margin = core.dpo_terms(pc, pr, rc, rr, .2)
    torch.testing.assert_close(margin, torch.tensor([.2, -.4]))
    torch.testing.assert_close(losses.mean(), -F.logsigmoid(torch.tensor([.2, -.4])).mean())
    assert accuracy.tolist() == [1., 0.]
    losses.mean().backward()
    assert pc.grad is not None and pr.grad is not None
    assert rc.grad is None and rr.grad is None


def test_identical_policy_reference_strict_accuracy_zero():
    values = torch.tensor([-2., -4.])
    losses, accuracy, margin = core.dpo_terms(values, values * 2, values, values * 2, .1)
    assert accuracy.eq(0).all() and margin.eq(0).all()
    torch.testing.assert_close(losses, torch.full_like(values, math.log(2)))


def test_small_update_learns_preference():
    tok = ToyTokenizer()
    policy, reference = ToyCausalLM(tok.vocab_size), ToyCausalLM(tok.vocab_size)
    optimizer = torch.optim.SGD(policy.parameters(), lr=.5)
    def loss():
        return core.compute_dpo_loss(policy, reference, tok, ["q"], ["a"], ["b"], .1, 8)
    before = float(loss().detach())
    loss().backward()
    optimizer.step()
    assert float(loss().detach()) < before
    assert all(p.grad is None for p in reference.parameters())
