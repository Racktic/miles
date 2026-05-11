# Miles + NanoRollout

This example launches Miles training on four H200 nodes while using an external NanoRollout HTTP service to run SWE-style digital-agent rollouts. Miles owns the policy model, training loop, SGLang inference engines, token/logprob capture, and PPO/GRPO update. NanoRollout owns the agent execution loop and environment lifecycle.

For NanoRollout background, see [NanoRollout: A Lightweight Infra for Digital Agent Rollout at Scale](https://ember-factory-33d.notion.site/NanoRollout-A-Lightweight-Infra-for-Digital-Agent-Rollout-at-Scale-312927eea9bd803792f4c3b954f8daa1?pvs=74).

## Quick Start

### Multi-Node Training

We originally ran this recipe on 4 H200 nodes. Start the NanoRollout server first, then run the launch script inside the Miles container on each training node:

```bash
MASTER_ADDR=<head-node-ip-or-hostname> NODE_RANK=<0-3> \
NANOROLLOUT_URL=http://<nanorollout-host>:11000 \
bash examples/nanorollout/run_qwen3_4b_instruct_4nodes.sh
```

`NODE_RANK=0` starts the Ray head, waits until all 32 GPUs are visible, and submits the Miles job. `NODE_RANK=1..3` join the Ray cluster as workers and stay alive until the head Ray process exits. If `NODE_RANK` is not set, the script uses `SLURM_NODEID`, then falls back to `0`.

## How Miles And NanoRollout Interact

The integration is implemented by `examples.nanorollout.generate_with_nanorollout.generate` and `examples.nanorollout.generate_with_nanorollout.generate_rollout`.

At each rollout step:

1. Miles samples prompts from the jsonl data and creates `Sample` objects.
2. The custom generate function builds a NanoRollout `/run` payload. The payload includes `instance_id`, `model_name`, `run_name`, `base_url`, `api_key`, `sampling_params`, `runner`, `task_type`, `env_type`, and agent timeout settings.
3. Miles posts that payload to `${NANOROLLOUT_URL}/run`.
4. NanoRollout runs the selected agent backend against the task environment. When the agent needs model tokens, it calls the OpenAI-compatible `base_url` supplied by Miles.
5. Because this script enables `--tito`, that `base_url` points to a per-process `TITOProxy` server created inside Miles. The proxy forwards generation to the Miles SGLang router and records exact tokens, response logprobs, loss masks, and optional routed experts.
6. NanoRollout returns messages, tools, reward, exit status, and agent metrics.
7. Miles converts the returned trajectory into training samples, applies the reward, logs aggregate agent metrics, and trains with GRPO.

## TITO Token Capture

TITO is the compatibility layer that lets an external agent framework call a normal OpenAI-compatible chat endpoint while Miles still trains on the exact tokens produced by SGLang. In this example, Miles creates one `TITOProxy` per rollout worker process. The proxy starts a small FastAPI server on a free local port and exposes `/v1/chat/completions`. When Miles builds the NanoRollout request, it passes this proxy URL as `base_url` and assigns a per-sample API key of the form `tito-<instance_id>-<sample_index>`. NanoRollout treats that endpoint like any other model provider.

For every chat completion call from the agent, `TITOProxy`:

1. Reads the OpenAI chat messages and tool schemas from NanoRollout.
2. Tokenizes only the newly added messages with the same tokenizer and chat template used by the training job.
3. Forwards generation to the Miles SGLang router through `/generate` with `return_logprob=true`.
4. Converts the SGLang output back to an OpenAI chat-completion response for the agent, including parsed tool calls when `--sglang-tool-call-parser qwen` is enabled.
5. Stores the exact generated token ids, response logprobs, and optional routed experts in a per-task `TaskState`.

After NanoRollout finishes the task, the custom generate function calls `proxy.get_task_result(task_id)` and writes the captured data directly onto the Miles `Sample`: `tokens`, `loss_mask`, `rollout_log_probs`, `rollout_routed_experts`, and `response_length`. This avoids reconstructing the training trajectory from text after the fact.

The loss mask is content-only. Prompt tokens, user/tool messages, ChatML assistant wrappers, and generated stop wrappers are masked out. Assistant content tokens receive loss. When function calling is enabled, malformed assistant turns without tool calls are also masked out so GRPO does not reinforce invalid tool formatting. If the agent fails before making any model call, Miles falls back to retokenizing the returned messages and assigns zero rollout logprobs for that fallback sample.

Per-sample metadata can override several agent-side fields:

| Metadata key | CLI fallback | Default |
| --- | --- | --- |
| `runner` | `--agent-runner` | `oh-lite` |
| `task_timeout` | `--agent-task-timeout` | `1800` |
| `step_timeout` | `--agent-step-timeout` | `600` |
| `eval_timeout` | `--agent-eval-timeout` | `600` |
| `env_timeout` | `--agent-env-timeout` | `120` |
| `create_timeout` | `--agent-create-timeout` | `600` |
| `max_iterations` | `--agent-max-iterations` | `100` |
| `use_fn_calling` | `--agent-no-use-fn-calling` | `true` |

The default launch script sets `--agent-runner oh-lite`, `--agent-max-iterations 35`, `--agent-task-timeout 6000`, `--agent-step-timeout 600`, `--agent-eval-timeout 600`, `--agent-env-timeout 120`, `--agent-create-timeout 600`, and `--agent-filter-overlong true`.

## Default Hyperparameters

The script follows the same layout as other Miles examples: `CKPT_ARGS`, `ROLLOUT_ARGS`, `EVAL_ARGS`, `OPTIMIZER_ARGS`, `GRPO_ARGS`, `PERF_ARGS`, `SGLANG_ARGS`, `AGENT_ARGS`, `MISC_ARGS`, and `CUSTOM_ARGS` are declared directly in `run_qwen3_4b_instruct_4nodes.sh`.

### Model And Checkpoints

| Argument | Default |
| --- | --- |
| `--model-name` | `Qwen3-4B-Instruct-2507` |
| `--hf-checkpoint` | `models/Qwen/Qwen3-4B-Instruct-2507` |
| `--ref-load` | `models/Qwen/Qwen3-4B-Instruct-2507` in bridge mode |
| `MEGATRON_LM_PATH` | `/root/Megatron-LM` |
| `MEGATRON_TO_HF_MODE` | `bridge` |

In bridge mode, actor/reference weights are loaded from the Hugging Face checkpoint through Megatron Bridge. If `MEGATRON_TO_HF_MODE=raw`, the reference checkpoint defaults to `${HF_CHECKPOINT}_torch_dist`.

Set `LOAD_PATH` to resume actor weights. Set `SAVE_PATH` to enable checkpoint saves, with `SAVE_INTERVAL` defaulting to `100`.

### Rollout And Data

| Argument | Default |
| --- | --- |
| `--prompt-data` | `examples/nanorollout/data/skyrl_v0_293.jsonl` |
| `--input-key` | `prompt` |
| `--metadata-key` | `metadata` |
| `--num-rollout` | `3000` |
| `--rollout-batch-size` | `64` |
| `--n-samples-per-prompt` | `16` |
| `--rollout-temperature` | `1.0` |
| `--rollout-max-context-len` | `65536` |
| `--rollout-max-response-len` | `4096` |
| `--global-batch-size` | `1024` |

Rollouts are shuffled and balanced with `--rollout-shuffle` and `--balance-data`. The health check waits up to 60 seconds before the first rollout.

### Evaluation

| Argument | Default |
| --- | --- |
| `--eval-interval` | `5` |
| `--eval-prompt-data` | `swe_bench_verified examples/nanorollout/data/swe_bench_verified.jsonl` |
| `--eval-input-key` | `prompt` |
| `--n-samples-per-eval-prompt` | `1` |

The script also sets `--skip-eval-before-train`.

### Optimization And GRPO

| Argument | Default |
| --- | --- |
| `--advantage-estimator` | `grpo` |
| `--use-kl-loss` | enabled |
| `--kl-loss-coef` | `0.01` |
| `--kl-loss-type` | `low_var_kl` |
| `--entropy-coef` | `0.0` |
| `--eps-clip` | `0.2` |
| `--eps-clip-high` | `0.28` |
| `--use-tis` | enabled |
| `--tis-clip` | `2.0` |
| `--tis-clip-low` | `0.0` |
| `--optimizer` | `adam` |
| `--lr` | `1e-6` |
| `--lr-decay-style` | `constant` |
| `--weight-decay` | `0.1` |
| `--adam-beta1` | `0.9` |
| `--adam-beta2` | `0.98` |

The optimizer uses CPU offload, overlapped CPU optimizer transfer, and the precision-aware optimizer.

### Parallelism And Performance

| Argument | Default |
| --- | --- |
| Nodes | `4` |
| GPUs per node | `8` |
| `--tensor-model-parallel-size` | `4` |
| `--pipeline-model-parallel-size` | `1` |
| `--context-parallel-size` | `2` |
| `--expert-model-parallel-size` | `1` |
| `--expert-tensor-parallel-size` | `1` |
| `--max-tokens-per-gpu` | `32768` |
| `--log-probs-chunk-size` | `8192` |
| `--rollout-num-gpus-per-engine` | `2` |
| `--sglang-context-length` | `65536` |
| `--sglang-mem-fraction-static` | `0.65` |
| `--sglang-tool-call-parser` | `qwen` |

## Runtime Environment Passed To Ray

The Ray job receives a runtime env containing:

| Variable | Purpose |
| --- | --- |
| `PYTHONPATH` | Adds Megatron-LM, this example directory, and the Miles repo. |
| `NANOROLLOUT_URL` | HTTP endpoint used by the custom generate function. |
| `SAVE_FAILED_TRAJ_DIR` | Directory for failed trajectory artifacts. |
| `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_API_KEY` | Optional W&B config. |
| `MASTER_ADDR` | Ray head address. |
| `CUDA_DEVICE_MAX_CONNECTIONS` | Set to `1` for Megatron. |
| `NCCL_NVLS_ENABLE`, `NCCL_TIMEOUT_MS` | NCCL behavior. |
