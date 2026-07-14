# Codebase Adaption TTT

Miles example for test-time co-training on Continual Learning Bench's
`codebase_adaptation` task.

This directory contains only training glue. The benchmark task, datasets,
containers, and scoring stay in `/home/qixinx/continual-learning-bench`.

The rollout follows the Alchemy memory co-training shape:

- ACT stream: solve one codebase issue using the current memory.
- WRITE stream: update memory after the issue.
- ACT reward: the benchmark instance reward.
- WRITE reward: downstream gain improvement, where `gain = reward - baseline_reward`.

Run from a Babel compute-node host with:

```bash
bash /home/qixinx/miles/examples/codebase_adaption/launch_codebase_adaption_apptainer.sh
```

The host launcher starts `miles_dev-202606081341.sif` and exposes the host
Apptainer runtime inside it, which clbench needs to start each issue SIF. The
inner Miles entrypoint remains `run_codebase_adaption_qwen3.5_4B.sh`.

SWE-Bench-CL execs intentionally omit `--fakeroot`. The outer Miles image's
fakeroot library requires GLIBC 2.38, while some official SWE-bench images use
older glibc versions. Their user-owned sandboxes remain writable without
fakeroot. The original `codebase_adaptation` defaults are unchanged.

The reproducible SWE-Bench-CL one-step smoke is:

```bash
bash /home/qixinx/miles/examples/codebase_adaption/run_swecl_smoke_qwen3.5_9B.sh
```

It uses two A100s with TP=2, two siblings, two issues per sibling, at most three
environment steps per issue, no heldout eval, and no checkpoint save.

By default, `CODEBASE_BASELINE_ARTIFACT` points at the merged SWE-Bench-CL and
codebase heldout baseline:

```text
/home/qixinx/miles/examples/codebase_adaption/data/baseline_merged.json
```

Override `CODEBASE_BASELINE_ARTIFACT` to compare against a different baseline.

The run script defaults to:

- HF checkpoint: `/data/user_data/qixinx/Qwen3.5-9B`
- torch_dist checkpoint: `/data/user_data/qixinx/Qwen3.5-9B_torch_dist`
- model spec: `scripts/models/qwen3.5-9B.sh`

For smoke tests, set `CODEBASE_NUM_ACTS_CAP`, `CODEBASE_MAX_STEPS_PER_ISSUE`,
to a small value. Set `CODEBASE_USE_WANDB=1`
to enable W&B logging when `WANDB_API_KEY` is available.

The default training data is `data/swecl_train_episodes.jsonl`: 54 fixed
SWE-Bench-CL episode templates, each containing a 9+10 blocked sequence over a
pair of repositories. Each Miles sample group selects one episode and its
siblings share the same instance order.

Heldout eval uses `data/heldout_episodes.jsonl`: five fixed `order_rank` values
from the heldout `3! * 3! = 36` repo-internal orders. Enable it with
`CODEBASE_EVAL_INTERVAL=<N>`.
