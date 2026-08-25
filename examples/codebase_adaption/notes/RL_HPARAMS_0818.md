# RL hyperparameters — codebase-adaptation runs (Qwen3.5-4B, miles/slime, SWE-smith)

Snapshot 2026-08-18. Sources: `run_codebase_adaption_qwen3.5_4B.sh`, `scripts/train_4b_v3nocurr_explore_gwin{,_think}.sh`, `codebase_advantage.py`, `codebase_rollout.py`, `codebase_config.yaml`. Both current runs share every value below; the only difference is the ACT-thinking flag (last table).

## Algorithm
| Item | Value |
|---|---|
| Policy optimizer | GRPO (critic-free, PPO-style clipped surrogate), strictly on-policy |
| Advantage | group-relative: (r − mean_group) / (std_group + 1e-6); group = 8 samples of the same prompt at the same episode position; ACT and WRITE samples grouped separately |
| Zero-std groups | dropped before the update (`CODEBASE_DROP_ZERO_STD_GROUPS=1`; ~40–50% of groups) |
| PPO epochs / grad steps per rollout | 1 (one gradient step per rollout batch, no sample reuse) |
| Clip ratio ε | 0.2 (low) / 0.28 (high) — asymmetric "clip-higher" (`--eps-clip 0.2 --eps-clip-high 0.28`) |
| Entropy coefficient | 0.0 |
| KL penalty in reward (`--kl-coef`) | 0.0 |
| KL loss to reference (`--use-kl-loss`) | on, coef 0.01, estimator k1; reference = frozen initial Qwen3.5-4B (`--ref-load`) |
| Loss aggregation | sequence-level policy loss (`--calculate-per-token-loss` off) |
| Importance-sampling correction (TIS) | off |

## Optimizer
| Item | Value |
|---|---|
| Optimizer | Adam (β1 = 0.9, β2 = 0.98) |
| Learning rate | 1e-6, constant (no warmup, no decay) |
| Weight decay | 0.1 |
| Precision | bf16 params/activations, fp32 grad all-reduce (`--accumulate-allreduce-grads-in-fp32`), attention softmax in fp32 |
| Dropout | 0.0 (attention & hidden) |

## Rollout / batching
| Item | Value |
|---|---|
| Prompts per RL step (`--rollout-batch-size`) | 2 (each prompt = one 12-issue episode over 2 repos, curriculum-ordered data) |
| Samples per prompt / group size (`--n-samples-per-prompt`) | 8 |
| Trajectories per step (`--global-batch-size`) | 16, dynamic global batch by token count (`--use-dynamic-batch-size`) |
| Total RL steps (`--num-rollout`) | 191 |
| Sampling temperature / top-p / top-k | 1.0 / 1.0 / off |
| Max new tokens per ACT turn (`--rollout-max-response-len`) | 2500 |
| Max new tokens per WRITE (memory rewrite) (`codebase_memory_max_tokens`) | 2048 |
| Max turns per issue (`CODEBASE_MAX_STEPS_PER_ISSUE`) | 40 |
| Sequence length cap for a packed multi-turn ACT trial (`--seq-length`) | 20480 tokens (think run) / 24576 (non-think run); over-long samples tail-truncated, or dropped if the prompt alone exceeds the cap |
| Max tokens per GPU for micro-batch packing (`--max-tokens-per-gpu`) | 20480 |
| Loss mask | assistant tokens only (thinking + answer + `<|im_end|>` in the think run) |

## Reward
| Item | Value |
|---|---|
| ACT reward (per issue) | 1 − (turns − 1)/40 if hidden tests pass, else 0 |
| WRITE reward (`gated_windowed`, K = 3) | format_ok × [mean(reward of next 3 issues) − mean(prev 3)] + 0.1 × format_ok |
| Exploration shaping on ACT | β = 0.3 × within-group standardized gpt-5-mini judge score of the memory delta, added to the advantage (raw reward untouched) |

## Infra
| Item | Value |
|---|---|
| Hardware | 1 node × 8 H100, colocated SGLang inference + Megatron training |
| Parallelism | TP = 2, PP = 1, CP = 1, sequence-parallel, full activation recompute |
| SGLang | 1 GPU per engine, mem fraction 0.5, server concurrency 512 |
| Eval / checkpoint interval | every 8 / 4 steps; heldout = 5 episodes × 6 issues (tablib + tenacity) |

## Runs
| Run | Difference | Status (2026-08-18) |
|---|---|---|
| `smith-4b-v3nocurr-explore-gwin` (K=3) | ACT non-thinking, seq-length 24576 | stopped at step 167 (format collapse; heldout 0.0); best heldout step 143; offline 19q: iter143 Cum Reward 3.750 / Cum Gain −0.150 |
| `smith-4b-v3nocurr-explore-gwin-think` | + `CODEBASE_ACT_THINKING=1` (`<think>` kept in context and trained), seq-length 20480 | step 78/191, heldout iter71 = 0.227 (run best) |
