# Frontier-CS group-memory training in Miles

This directory contains the Frontier-CS rollout and training integration for
Miles. It is intentionally separate from the existing codebase-adaptation task.

The normal setup uses two Git repositories:

- the Miles branch contains the training code and the problem-group JSONL;
- the Frontier-CS branch contains problem statements, hidden test data, and the
  judge implementation.

There is no separate dataset checkout step. Clone both repositories at the
revisions used by the experiment.

## 1. Clone the two repositories

Clone the two published `frontiercs-ttt` branches:

```bash
git clone --branch frontiercs-ttt https://github.com/Racktic/miles.git \
  /home/your-user/miles
git clone --branch frontiercs-ttt https://github.com/Racktic/Frontier-CS.git \
  /home/your-user/Frontier-CS

export MILES_ROOT=/home/your-user/miles
export FRONTIERCS_ROOT=/home/your-user/Frontier-CS
```

For an exact commit rather than a branch, clone the repository normally and
then run `git checkout <commit>` in that repository.

## 2. Training data

Both required data inputs are already supplied by the two clones:

| Input | Location | Purpose |
|---|---|---|
| Group dataset | `${MILES_ROOT}/examples/frontiercs_ttt/data/problem_groups_30.jsonl` | Thirty groups, three problem IDs per group |
| Benchmark data | `${FRONTIERCS_ROOT}/algorithmic/problems` | Statements, configs, checkers, and hidden test cases |

The group JSONL is small and is committed to the Miles branch. The benchmark
data is committed to the Frontier-CS branch. No Hugging Face dataset download
is required for this task.

Verify both inputs before starting a GPU allocation:

```bash
test -f "${MILES_ROOT}/examples/frontiercs_ttt/data/problem_groups_30.jsonl"
test -d "${FRONTIERCS_ROOT}/algorithmic/problems"
```

Use `FRONTIERCS_PROMPT_DATA` only when intentionally selecting a different
group file, such as the smoke-test dataset.

## 3. Install Miles and prepare the model

Use the normal Miles environment or container described by the top-level Miles
documentation. For a source installation:

```bash
cd "${MILES_ROOT}"
pip install -r requirements.txt
pip install -e .
```

The launcher requires two copies of the model:

1. the Hugging Face checkpoint used by the rollout engine;
2. the Megatron `torch_dist` checkpoint used by the trainable actor and
   reference model.

For Qwen3.5-4B:

```bash
export FRONTIERCS_MODEL_ROOT=/home/your-user/models
export MEGATRON_LM_PATH=/home/your-user/Megatron-LM

hf download Qwen/Qwen3.5-4B \
  --local-dir "${FRONTIERCS_MODEL_ROOT}/Qwen3.5-4B"

cd "${MILES_ROOT}"
source scripts/models/qwen3.5-4B.sh
PYTHONPATH="${MEGATRON_LM_PATH}" python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${FRONTIERCS_MODEL_ROOT}/Qwen3.5-4B" \
  --save "${FRONTIERCS_MODEL_ROOT}/Qwen3.5-4B_torch_dist"
```

Do not commit model weights to either repository.

For Qwen3.6-27B, use the corresponding Miles model configuration during
conversion:

```bash
export FRONTIERCS_MODEL_ROOT=/home/your-user/models
export MEGATRON_LM_PATH=/home/your-user/Megatron-LM

hf download Qwen/Qwen3.6-27B \
  --local-dir "${FRONTIERCS_MODEL_ROOT}/Qwen3.6-27B"

cd "${MILES_ROOT}"
source scripts/models/qwen3.6-27B.sh
PYTHONPATH="${MEGATRON_LM_PATH}" python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${FRONTIERCS_MODEL_ROOT}/Qwen3.6-27B" \
  --save "${FRONTIERCS_MODEL_ROOT}/Qwen3.6-27B_torch_dist"
```

## 4. Configure the judge

Training communicates with the judge over HTTP. The only value required by the
rollout workers is `FRONTIERCS_JUDGE_URL`.

### Recommended: independent Docker judge

Run the judge on a host that supports privileged Docker:

```bash
cd "${FRONTIERCS_ROOT}/algorithmic"
docker compose up --build -d
curl -fsS http://127.0.0.1:8081/health
```

Then set `FRONTIERCS_JUDGE_URL` to the address reachable from every Ray worker.
If the GPU cluster does not allow privileged containers, run this service on a
separate machine and expose its port only to the training cluster.

The Docker image installs Node.js, C++ tooling, and the pinned go-judge runtime.
It mounts `${FRONTIERCS_ROOT}/algorithmic/problems`, so it evaluates the test
data from the checked-out Frontier-CS revision.

### Fallback: host judge on the training node

Use this only when the compute node supports the cgroup/namespace operations
required by go-judge. Install Node.js 20, C++17, go-judge, and go-judge-init,
then install the Node dependencies:

```bash
cd "${FRONTIERCS_ROOT}/algorithmic"
npm ci
```

Set `FRONTIERCS_AUTO_START_JUDGE=1`. If the executables are not on `PATH`, also
set `FRONTIERCS_GOJUDGE_BIN`, `FRONTIERCS_GOJUDGE_INIT`, and
`FRONTIERCS_NODE_BIN`. The launcher uses `host_judge.py` to start and stop an
isolated local service; it does not modify the Frontier-CS checkout.

## 5. Configure the run

Copy the environment template to a private machine-specific file and edit it:

```bash
cp "${MILES_ROOT}/examples/frontiercs_ttt/frontiercs_env.example" \
  /home/your-user/frontiercs.env
```

The required paths are:

| Variable | Meaning |
|---|---|
| `FRONTIERCS_ROOT` | Frontier-CS repository clone |
| `FRONTIERCS_HF_CHECKPOINT` | Hugging Face model directory |
| `FRONTIERCS_TORCH_DIST` | Converted Megatron checkpoint directory |
| `FRONTIERCS_OUTPUT_ROOT` | Rollout traces and environment state |
| `FRONTIERCS_SAVE_DIR` | Training checkpoints |
| `FRONTIERCS_JUDGE_URL` | Healthy judge HTTP endpoint |

The current full experiment settings are:

| Variable | Default | How to choose it |
|---|---:|---|
| `FRONTIERCS_GROUP_SIZE` | `3` | Must match the number of problems in every JSONL group |
| `FRONTIERCS_GROUPS_PER_UPDATE` | `2` | Complete group episodes collected before one optimizer step |
| `FRONTIERCS_MEMORY_ROUNDS` | `4` | Number of solve/evaluate rounds in each episode |
| `FRONTIERCS_CANDIDATES_PER_PROBLEM` | `1` | Independent candidates per problem per round |
| `FRONTIERCS_TRAIN_WRITE` | `1` | Set to `0` to generate/use memory but exclude WRITE responses from optimization |
| `FRONTIERCS_ACT_MAX_NEW_TOKENS` | `25600` | Maximum problem-solving response length |
| `FRONTIERCS_WRITE_MAX_NEW_TOKENS` | `25600` | Maximum memory response length |
| `FRONTIERCS_SEQ_LENGTH` | `32768` | Maximum trainable prompt-plus-response sequence |
| `FRONTIERCS_KL_MODE` | `loss` | Use `loss` for KL loss or `none` to disable it |
| `FRONTIERCS_KL_COEF` | `0.01` | KL-loss coefficient |
| `FRONTIERCS_KL_TYPE` | `k1` | Miles KL estimator: `k1`, `k2`, `k3`, or `low_var_kl` |
| `FRONTIERCS_LR` | `1e-6` | Full-parameter actor learning rate |

Keep `FRONTIERCS_GROUP_SIZE=3` for the supplied 30-group dataset. The default
`FRONTIERCS_GROUPS_PER_UPDATE=2` means that one epoch over 30 groups contains 15
optimizer steps. Increase this value only when rollout time and memory permit.

Hardware parameters are cluster-specific:

| Variable | Default |
|---|---|
| `FRONTIERCS_NGPU` | Number of visible GPUs |
| `FRONTIERCS_TP` | `FRONTIERCS_NGPU` |
| `FRONTIERCS_PP` | `1` |
| `FRONTIERCS_CP` | `1` |
| `FRONTIERCS_ACTOR_NUM_NODES` | `1` |
| `FRONTIERCS_ACTOR_GPUS_PER_NODE` | `FRONTIERCS_NGPU` |
| `FRONTIERCS_ROLLOUT_NUM_GPUS` | Total actor GPUs |
| `FRONTIERCS_ROLLOUT_GPUS_PER_ENGINE` | `1` |
| `FRONTIERCS_SGLANG_MEM_FRACTION` | `0.5` |

Before submitting training, the launcher waits until Ray contains the requested
number of nodes and GPUs. An incomplete allocation therefore fails with an
explicit timeout instead of hanging later while Miles creates its placement
group. Set `FRONTIERCS_WAIT_FOR_RAY_CLUSTER=0` only when readiness is validated
externally. By default it also verifies the repositories, data, model inputs,
and output directories on every eligible GPU node.

W&B is optional. Set `WANDB_PROJECT=miles-frontier-cs` and provide the API key
through the process environment or `FRONTIERCS_WANDB_ENV_FILE`. Never commit the
key. Set `FRONTIERCS_USE_WANDB=0` to disable logging.

## 6. Launch

Enter the GPU allocation, activate the Miles environment, source the private
configuration, and run the complete-episode launcher:

```bash
source /home/your-user/frontiercs.env
cd "${MILES_ROOT}"
bash examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.5_4B.sh
```

For Qwen3.6-27B, set its two checkpoint paths and use the model-specific
entrypoint. It defaults to `TP=4`; PP, CP, and the node counts remain explicit
hardware parameters:

```bash
export FRONTIERCS_HF_CHECKPOINT=/home/your-user/models/Qwen3.6-27B
export FRONTIERCS_TORCH_DIST=/home/your-user/models/Qwen3.6-27B_torch_dist
bash /home/your-user/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.6_27B.sh
```

The launcher verifies the group dataset, model directories, visible GPUs, and
judge health before submitting the Ray job. Slurm directives, container mounts,
and cluster module commands belong in a small cluster-specific wrapper rather
than in the shared launcher.

### Multi-node launch

`run_frontiercs_ttt_multinode.sh` is the scheduler-independent multi-node
entrypoint. Every process uses the same head address. Set the total number of
nodes and per-node GPUs on every node before starting it:

```bash
export FRONTIERCS_ACTOR_NUM_NODES=4
export FRONTIERCS_ACTOR_GPUS_PER_NODE=8
export FRONTIERCS_NGPU=8
export FRONTIERCS_TP=8
export FRONTIERCS_PP=2
export FRONTIERCS_CP=1
```

Start the head role on the designated head node:

```bash
source /home/your-user/frontiercs.env
bash /home/your-user/miles/examples/frontiercs_ttt/run_frontiercs_ttt_multinode.sh \
  head 10.0.0.10
```

The Qwen3.6-27B entrypoint accepts the same role and address, so it can be used
directly on the head:

```bash
bash /home/your-user/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.6_27B.sh \
  head 10.0.0.10
```

Start one worker role on each of the other nodes:

```bash
source /home/your-user/frontiercs.env
bash /home/your-user/miles/examples/frontiercs_ttt/run_frontiercs_ttt_multinode.sh \
  worker 10.0.0.10
```

Use the matching Qwen3.6-27B entrypoint on every worker when training that
model:

```bash
bash /home/your-user/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.6_27B.sh \
  worker 10.0.0.10
```

Workers wait for the Ray head and monitor it after joining. The head waits until
all four nodes advertise eight GPUs each and is the only process that submits
the training job. When that process finishes or fails, it stops the Ray head;
workers detect the closed head and clean up their local Ray services. Miles
receives 32 colocated actor and rollout GPUs. The rollout manager is pinned to
the Ray head so the Frontier-CS episode state and a head-local judge have a
single, deterministic owner.

For a Ray cluster that the scheduler has already created, invoke the normal
launcher with `FRONTIERCS_START_RAY=0`, set the dashboard endpoint in
`FRONTIERCS_RAY_ADDRESS`, and set the GCS endpoint in
`FRONTIERCS_RAY_CLUSTER_ADDRESS`. The same resource-readiness check runs before
job submission.

The launcher also verifies that the samples in one optimizer update are exactly
divisible across data-parallel ranks. Each default `G=3`, `K=1`, `S=4` episode
contains 15 trainable samples, so the check is
`FRONTIERCS_GROUPS_PER_UPDATE * 15` divisible by
`DP = total_GPUs / (TP * PP * CP)`. It fails before allocating model actors
rather than letting Miles silently trim complete-episode samples. TP, PP, or
the number of groups per update can be adjusted to satisfy this invariant.

## 7. Training semantics

One optimizer unit is a batch of complete group episodes. Model weights remain
frozen while one group passes through all memory rounds:

1. In each round, generate `G*K` solutions from the current shared memory.
2. Submit every extracted C++ solution to the Frontier-CS judge.
3. Give the previous memory plus candidate code and evaluator feedback to one
   memory-generation call.
4. Pass that memory to the next round. There is no fallback to the previous
   memory when the generated memory is empty or invalid.
5. After all `S` rounds finish, return the ACT samples and the `S-1` WRITE
   samples whose downstream rewards are observable, then update the model.

With `G=3`, `K=1`, `S=4`, one episode contains 12 problem-solving samples and 3
memory samples. With two episodes per update, an optimizer step receives 30
samples.

For a solution-only optimization ablation, set `FRONTIERCS_TRAIN_WRITE=0`. The
pipeline still generates the same nonterminal memories and supplies them to
later rounds; it simply returns no WRITE samples to Miles. With the same
`G=3`, `K=1`, `S=4`, each episode then contains 12 trainable samples.

Problem-solving advantages use `temporal_problem_relative`: the `S*K` rewards
for the same problem within an episode are standardized by their mean and sample
standard deviation. Memory advantages use the downstream group-score delta.
Memory samples are not normalized together with problem-solving samples.

## 8. Outputs and recovery

All inspectable rollout state is stored under
`${FRONTIERCS_OUTPUT_ROOT}/${FRONTIERCS_RUN_ID}`. Model checkpoints are stored
under `FRONTIERCS_SAVE_DIR`.

Each episode records its prompts, raw generations, extracted reasoning, C++
code, evaluator feedback, memory, packed training samples, and an atomic episode
commit. A completed episode is replayed from disk after restart instead of being
sampled and judged again. Do not reuse a run ID with a different group, round,
or candidate configuration.

The W&B dashboard logs numeric score, execution, failure, response-length,
memory-length, memory-change, and training-signal metrics. It does not upload
prompts, code, diagnostics, or memory text.
