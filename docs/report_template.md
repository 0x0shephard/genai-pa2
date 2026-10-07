# Instruction Tuning and Preference Optimization of Small Language Models

**Author:** TODO · **GitHub:** TODO · **Maximum final length: 8 pages including references**

## 1. Repository, models, and data

Qwen/Qwen2.5-0.5B is the base for SFT and both independent full-parameter DPO tracks. SmolLM2-360M-Instruct is the persona baseline. Report actual source revisions, filtered subset counts, validation counts, and seed from `configurations.json`. Document the single-turn conversions and prompt caps. Exact prepared subsets and manifests are included in saved Kaggle outputs.

TODO: Link the submitted repository, notebook versions, and artifacts. State the fixed AlpacaEval and HH set counts and persona prompt categories.

## 2. Experimental configuration

| Run | Strategy | Learning rate | Schedule / warmup | Batch × accumulation | Context | Epochs | Beta | Seed |
|---|---|---|---|---|---|---|---|---|
{{CONFIGURATION_TABLE}}

TODO: Add optimizer (AdamW), weight decay, clipping, precision, actual optimizer steps, exact reference checkpoints, and generation settings. For LoRA include alpha, dropout, and target modules. The JSON evidence includes these values; report submitted runs, not just intended defaults.

## 3. Parameter choices and deviations

SFT uses response-token mean NLL. Each DPO track sums response log probabilities against the same frozen SFT checkpoint. Persona LoRA uses rank 16, alpha 32, and attention projections to limit trainable parameters while steering style. Its learning rate and three epochs are an initial recipe, not evidence of optimality.

TODO: Explain each meaningful change relative to the manual, distinguishing memory-driven changes from quality-driven changes. Support changes with pilot results, actual memory behavior, validation diagnostics, or output examples. Do not claim improvement without measurements.

## 4. Training behavior

Insert compact figures from `figures/`. They show raw metrics without smoothing; dashed lines identify selected checkpoints. SFT selection uses validation loss. DPO selection uses strict reward accuracy with validation loss as a tie-breaker; persona selection uses validation reward accuracy.

TODO: Explain convergence, overfitting, instabilities, and selection. Identical policy/reference log probabilities imply initial strict reward accuracy zero; fp16 reference rounding can produce small nonzero margins in real runs. Do not equate training loss with final response quality.

## 5. DPO-versus-baseline results

Wins refer to the candidate DPO model over its baseline. Unknown judgments and failures are excluded from both win-rate denominators and reported separately. Decisive rate = wins / (wins + losses). Tie-adjusted rate = (wins + 0.5 × ties) / (wins + ties + losses).

| Track / dimension | N | Wins | Ties | Losses | Unknown | Failures | Decisive rate | Tie-adjusted rate |
|---|---|---|---|---|---|---|---|---|
{{RESULTS_TABLE}}

TODO: Report judge model, request/retry counts, failed IDs, and generation settings. Keep persona style and quality separate, and do not pool unlike evaluation sets.

## 6. Interpretation and error analysis

TODO: State which model performed better on each set, by how much, and plausible evidence-backed reasons. Cover refusals, factuality, instruction following, verbosity, and style where relevant. Select representative improvements and regressions from `qualitative_examples.json`; manually verify their relevance and factual claims.

## 7. Limitations and reproducibility

TODO: Discuss model-judge bias, sample size, single-seed variance, filtering, truncated prompts/responses, and compute constraints. List repository and Kaggle paths for configs, checkpoints, metrics, prompts, generations, comparisons, and judgments. Identify incomplete or failed runs explicitly.

## References

- [Assignment starter and manual](https://github.com/ahmad903501/small-model-alignment-starter).
- [Rafailov et al., Direct Preference Optimization (2023)](https://arxiv.org/abs/2305.18290).
- [UltraChat 200k dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k).
- [Anthropic HH-RLHF dataset card](https://huggingface.co/datasets/Anthropic/hh-rlhf).
- [UltraFeedback binarized dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval).
- [TRL DPOTrainer 0.15.2](https://huggingface.co/docs/trl/v0.15.2/dpo_trainer) and [PEFT LoRA](https://huggingface.co/docs/peft/v0.14.0/en/package_reference/lora).
- [Kaggle Benchmarks SDK](https://github.com/Kaggle/kaggle-benchmarks).
