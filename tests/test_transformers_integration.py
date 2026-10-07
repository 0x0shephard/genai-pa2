"""Optional API integration checks with randomly initialized tiny models."""
import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from alignment import core
from tests.toy_lm import ToyTokenizer


def test_actual_qwen_forward_causal_loss_and_reference_gradients():
    tok = ToyTokenizer()
    config = transformers.Qwen2Config(vocab_size=tok.vocab_size, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, use_sliding_window=False)
    policy = transformers.Qwen2ForCausalLM(config)
    reference = copy.deepcopy(policy).eval()
    batch = core.build_sft_batch(tok, ["ab", "c"], ["yes", "no"], 16)
    scores = core.response_logps_from_batch(policy, batch)
    actual = policy(**batch).loss
    torch.testing.assert_close(-scores.sum() / core.response_token_count(batch), actual)
    loss = core.compute_dpo_loss(policy, reference, tok, ["a"], ["yes"], ["no"], .1, 16)
    torch.testing.assert_close(loss.detach(), torch.tensor(2.).log())
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())
    assert all(p.grad is None for p in reference.parameters())


def test_pinned_trl_lora_tiny_cpu_step_and_checkpoint(tmp_path):
    trl = pytest.importorskip("trl")
    peft = pytest.importorskip("peft")
    datasets = pytest.importorskip("datasets")
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "question": 3, "good": 4, "bad": 5}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend,
        pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    config = transformers.LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        eos_token_id=1, pad_token_id=0)
    model = transformers.LlamaForCausalLM(config)
    initial_embedding = model.model.embed_tokens.weight.detach().clone()
    data = datasets.Dataset.from_list([{"prompt": "question", "chosen": "good", "rejected": "bad"}] * 4)
    args = trl.DPOConfig(output_dir=str(tmp_path / "trainer"), use_cpu=True, fp16=False,
        max_steps=1, per_device_train_batch_size=2, gradient_accumulation_steps=2,
        per_device_eval_batch_size=2, max_length=16, max_prompt_length=8,
        eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
        metric_for_best_model="eval_rewards/accuracies", greater_is_better=True,
        disable_dropout=False, report_to="none", logging_steps=1,
        learning_rate=.01, lr_scheduler_type="cosine", remove_unused_columns=False)
    trainer = trl.DPOTrainer(model=model, args=args, processing_class=tok,
        train_dataset=data, eval_dataset=data,
        peft_config=peft.LoraConfig(r=2, lora_alpha=4, lora_dropout=.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    trainer.train()
    assert trainer.state.best_model_checkpoint
    assert "eval_rewards/accuracies" in trainer.evaluate()
    torch.testing.assert_close(trainer.model.get_base_model().model.embed_tokens.weight, initial_embedding)
    path = tmp_path / "adapter"
    trainer.save_model(str(path))
    assert (path / "adapter_config.json").exists()
