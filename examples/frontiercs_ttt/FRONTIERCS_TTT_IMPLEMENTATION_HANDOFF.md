# Frontier-CS TTT: Complete Implementation and Experiment Handoff

This document is the implementation-level source of truth for the current
Frontier-CS shared-memory training pipeline in Miles. It is written for an
engineer or coding agent taking over experiments without relying on prior chat
history.

The short README explains installation and launch. This document explains the
research formulation, exact data flow, code paths, reward and advantage math,
trace format, distributed execution, invariants, failure handling, validation,
and the operating procedure for new experiments.

## 0. Current status and scope

### What is implemented

- Complete multi-round problem-group episodes are sampled before an optimizer
  update.
- The same frozen policy version is used for every solution and memory
  generation within an episode.
- A group-shared memory is updated from candidate code and evaluator feedback.
- Task reward and delayed memory reward are packed into separate training
  samples.
- Task advantages can be normalized across rounds for each problem.
- An optional external-LLM exploration reward compares only the memory before
  and after a round.
- When exploration reward is enabled, a terminal memory is generated after the
  final solution round so the final solutions also receive exploration credit.
- Scalar W&B metrics, durable rollout traces, deterministic retry/replay, and
  multi-node Ray startup are implemented.
- Qwen3.5-4B and Qwen3.6-27B model entrypoints are present.
- An explicit `FRONTIERCS_TRAIN_WRITE` switch supports solution-only
  optimization while preserving memory generation and cross-round use.

### What is not yet finished or empirically validated

- The exploration-judge rubric is implemented but is still a provisional
  prompt. It has not yet been calibrated on a labeled set of real memory pairs.
- No live external-API plus GPU one-step training smoke test has been completed
  for the new exploration-reward path.
- The coefficient `FRONTIERCS_ACT_EXPLORE_BETA` has not been tuned. Its default
  remains `0`, which completely disables external exploration judging.
- The training rollout currently uses the default Frontier-CS prompt profile
  directly. A prompt-version environment/CLI switch is not yet threaded through
  the training code.
- The external memory judge is a coarse group-round critic. It cannot attribute
  a memory improvement to an individual problem or candidate.

Do not report exploration-reward training as validated until the calibration
and live smoke-test checkpoints in Section 18 have passed.

## 1. Repository contract

The system deliberately spans two repositories.

| Repository | Local checkout in the current workspace | Responsibility |
|---|---|---|
| Miles | `/home/qixinx/miles` | Rollout orchestration, reward/advantage calculation, model training, Ray launch, W&B metrics, training tests |
| Frontier-CS | `/home/qixinx/Frontier-CS` | Problem statements, hidden test data, judge service, prompt construction, C++ extraction, shared trace types |

The intended remote branches are named `frontiercs-ttt` in both repositories.
The current Miles branch is `frontiercs-ttt`. The last pushed Miles base commit
before the uncommitted exploration-reward changes was
`813206def0cf19ec4f489842bed330317060c2de`.

At the time this handoff was written, the exploration-reward implementation and
this document are working-tree changes. Before handing a run to another cluster:

1. Inspect `git status` in both repositories.
2. Commit only the intended Frontier-CS files. There are unrelated pre-existing
   working-tree changes under `/home/qixinx/miles/examples/codebase_adaption` and
   they must not be included in a Frontier-CS commit.
3. Push both `frontiercs-ttt` branches.
4. Record the exact two commit hashes in the experiment record.

The normal portable layout is two sibling clones under one `/home/...`
directory. `FRONTIERCS_ROOT` can override discovery. There is intentionally no
`FRONTIERCS_WORKSPACE` variable and no separate checkout helper.

## 2. Research formulation

### 2.1 Objective

The goal is not held-out IID generalization. The goal is to raise performance
on a selected collection of hard Frontier-CS optimization problems while
learning a shared, higher-level memory that transfers useful search knowledge
between related problems and between successive attempts.

Problem overlap between dataset groups is intentional. The training code treats
the JSONL as an ordinary dataset; it does not deduplicate problems and does not
perform a train/held-out split.

### 2.2 Symbols

| Symbol | Meaning | Current default |
|---|---|---:|
| `N` | Number of group records in the dataset | `30` |
| `G` | Problems in one group | `3` |
| `K` | Candidate solutions per problem per round | `1` |
| `S` | Solution/memory rounds in one complete episode | `4` |
| `B` | Complete group episodes collected per optimizer update | `2` |
| `M_t` | Shared memory supplied to solution round `t` | `M_0` is empty |
| `C_{t,p,k}` | Candidate for round `t`, problem `p`, sample `k` | one C++ response |
| `R_{t,p,k}` | Bounded task reward | judge score divided by `100` |
| `Q_t` | Mean bounded score of the whole group in round `t` | in `[0,100]` |
| `E_t` | Raw memory-delta exploration score | in `[0,1]` |

### 2.3 Non-negotiable episode semantics

For one group record, all `S` rounds are generated before Miles performs an
optimizer update. There is no update between rounds. This guarantees that a
four-round temporal comparison is based on one frozen policy version.

In each round:

1. Every problem sees the same current memory `M_t`.
2. The solution prompt contains the current problem statement and `M_t`.
3. By default it does **not** contain raw previous diagnostics, previous
   reasoning, or previous code for that problem.
4. The `G*K` candidates are generated concurrently.
5. Every extracted program is evaluated by the Frontier-CS judge.
6. After a barrier, one memory response is generated from `M_t`, all current
   problem statements, all current candidate code, and evaluator feedback.
7. The memory response becomes `M_{t+1}` exactly as generated after light fence
   cleanup. There is no fallback to `M_t` when the response is empty or poor.

The full default timeline is:

```text
frozen policy theta_i

M0 = empty
  -> round 0: 3 problems x 1 candidate -> judge -> Q0 -> write M1
  -> round 1: 3 problems x 1 candidate -> judge -> Q1 -> write M2
  -> round 2: 3 problems x 1 candidate -> judge -> Q2 -> write M3
  -> round 3: 3 problems x 1 candidate -> judge -> Q3
       -> if exploration beta > 0: write terminal M4 for E3 only

return 12 solution samples + 3 trainable memory samples
  -> calculate rewards and advantages
  -> one full-parameter optimizer update to theta_(i+1)
  -> publish theta_(i+1) to rollout engines
```

With exploration disabled, `M4` is not generated. With exploration enabled,
`M4` is generated and traced but is not a training sample.

`FRONTIERCS_TRAIN_WRITE=0` changes only which samples enter the optimizer. The
writer is still called after rounds `0 ... S-2`, its exact output still becomes
the next round's memory, and its trace/metrics remain available. With positive
exploration beta, the terminal writer call is also unchanged. No WRITE response
is packed as a training sample and no delayed WRITE reward is computed.

## 3. Dataset and group construction

The canonical dataset is
`/home/qixinx/miles/examples/frontiercs_ttt/data/problem_groups_30.jsonl`.
The smoke dataset is
`/home/qixinx/miles/examples/frontiercs_ttt/data/problem_groups_smoke.jsonl`.

Each non-empty JSONL row has this shape:

```json
{
  "prompt": [{"role": "user", "content": "Frontier-CS related problem group"}],
  "metadata": {
    "group_id": "color_sat_bridge",
    "problem_ids": ["174", "175", "177"],
    "high_level_family": "coloring, SAT reduction, and incremental constraint repair"
  }
}
```

The generic `prompt` content is only a Miles dataset carrier. The real solution
prompts are built from `metadata.problem_ids` and the corresponding benchmark
statements.

The rollout validates that:

- `group_id` is present;
- `problem_ids` is non-empty;
- no problem is duplicated within one group;
- the number of IDs equals `FRONTIERCS_GROUP_SIZE`.

The 30 current groups are:

| # | Group | Problems | Intended shared structure |
|---:|---|---|---|
| 1 | `color_sat_bridge` | 174, 175, 177 | coloring, SAT reduction, incremental constraint repair |
| 2 | `sat_color_bridge` | 176, 177, 178 | SAT-to-color reduction and search across scale regimes |
| 3 | `cut_sat_bridge` | 174, 192, 193 | binary labeling, Max-Cut, exact Cut-to-2SAT structure |
| 4 | `graph_permutation` | 180, 181, 313 | permutation optimization and incremental swap deltas |
| 5 | `facility_location` | 301, 307, 308 | facility placement, assignment, marginal selection |
| 6 | `network_investment` | 306, 308, 309 | fixed setup cost and repeated downstream network demand |
| 7 | `sequence_adjacency` | 302, 305, 313 | ordering, adjacency costs, incremental local moves |
| 8 | `walk_coverage` | 303, 304, 312 | walk construction, coverage, route balancing |
| 9 | `structured_intervention` | 305, 310, 314 | structured interventions and global marginal effects |
| 10 | `weighted_selection` | 301, 310, 315 | budgeted representative selection and weighted coverage |
| 11 | `color_sat_scale` | 174, 175, 176 | verified color-to-SAT reduction and SAT scale transfer |
| 12 | `sat_color_reduction` | 175, 176, 177 | SAT local search and verified SAT-to-color reduction |
| 13 | `sat_density_scale` | 175, 176, 178 | Max-3-SAT across size and density regimes |
| 14 | `sat_clause_arity` | 175, 178, 193 | SAT search across scale and clause arity |
| 15 | `color_proper_bridge` | 174, 177, 186 | coloring across scale and soft versus proper constraints |
| 16 | `color_clique_cover` | 174, 186, 187 | coloring and coloring-to-clique-cover duality |
| 17 | `sat_cut_reduction` | 175, 192, 193 | SAT search and exact Max-Cut-to-Max-2-SAT reduction |
| 18 | `cover_mis_scale` | 182, 183, 184 | vertex-cover and independent-set complement structure |
| 19 | `cover_mis_clique` | 182, 183, 185 | cover-to-independent-set and complement-clique transformations |
| 20 | `mis_complement_clique` | 183, 184, 185 | independent-set scale transfer and complement clique search |
| 21 | `independent_coloring` | 184, 186, 187 | independent sets as coloring and clique-cover primitives |
| 22 | `clique_cover_primitive` | 185, 186, 187 | clique search as a coloring and clique-cover primitive |
| 23 | `independent_clique_partition` | 183, 185, 187 | independent sets, cliques, and clique partition search |
| 24 | `graph_qap_placement` | 180, 181, 311 | graph matching, QAP, and pairwise placement |
| 25 | `qap_permutation` | 181, 311, 313 | pairwise permutation objectives and swap-delta caches |
| 26 | `adjacency_periodic_permutation` | 302, 311, 313 | adjacency, periodic placement, and permutation local search |
| 27 | `schedule_permutation` | 305, 311, 313 | scheduling and permutation local search |
| 28 | `budgeted_facility` | 301, 308, 310 | selection under cardinality and budget constraints |
| 29 | `budgeted_saturation` | 308, 310, 315 | budgeted selection, marginal gains, saturation |
| 30 | `network_intervention` | 306, 308, 314 | limited setup interventions affecting many downstream queries |

Problem titles can be read from
`/home/qixinx/Frontier-CS/algorithmic/problems/<problem-id>/statement.txt`.
The 30 rows contain repeated memberships by design. Do not “fix” the overlap
unless the experiment explicitly ablates it.

## 4. Code map and call chain

### 4.1 Miles-side files

| File | Role |
|---|---|
| `/home/qixinx/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.5_4B.sh` | Canonical launcher and validation for complete-episode training |
| `/home/qixinx/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.6_27B.sh` | Qwen3.6-27B model wrapper; defaults to TP=4, then calls the canonical launcher |
| `/home/qixinx/miles/examples/frontiercs_ttt/run_frontiercs_ttt_multinode.sh` | Scheduler-independent Ray head/worker wrapper |
| `/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_episode_rollout.py` | Canonical complete frozen-policy episode generator |
| `/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_rollout.py` | Shared helper functions plus an older round-wise formulation |
| `/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_advantage.py` | Task, memory, and exploration advantage calculation |
| `/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_exploration_judge.py` | External LLM memory-delta judge |
| `/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_metrics.py` | Numeric rollout metrics and W&B logging |
| `/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_episode_config.yaml` | Custom rollout defaults exposed to Miles |
| `/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_env.example` | Portable environment template |
| `/home/qixinx/miles/examples/frontiercs_ttt/host_judge.py` | Optional local judge lifecycle manager |
| `/home/qixinx/miles/examples/frontiercs_ttt/wait_for_ray_cluster.py` | Multi-node resource and shared-path readiness check |
| `/home/qixinx/miles/examples/frontiercs_ttt/tests` | Unit/integration-style tests with mocked inference and judging |

`/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_rollout.py` contains the
older update-between-rounds entrypoint. It is retained for reproducibility, but
new canonical experiments must use `frontiercs_episode_rollout.generate_episode`.

### 4.2 Frontier-CS-side files

| File | Role |
|---|---|
| `/home/qixinx/Frontier-CS/qwen_eval/frontiercs_ttt/prompts.py` | Builds solution and memory prompts; extracts the final C++ block |
| `/home/qixinx/Frontier-CS/qwen_eval/frontiercs_ttt/prompt_versions/registry.py` | Prompt profile registry; current default is `qwen35_current` |
| `/home/qixinx/Frontier-CS/qwen_eval/frontiercs_ttt/prompt_versions/qwen35_current` | Current solution and memory prompt fragments |
| `/home/qixinx/Frontier-CS/qwen_eval/frontiercs_ttt/judge.py` | Async HTTP judge adapter and feedback projection |
| `/home/qixinx/Frontier-CS/qwen_eval/frontiercs_ttt/trace.py` | Atomic trace writer and path layout |
| `/home/qixinx/Frontier-CS/qwen_eval/frontiercs_ttt/types.py` | `ProblemSpec`, `CandidateRecord`, `JudgeFeedback`, and `ModelReply` |
| `/home/qixinx/Frontier-CS/algorithmic` | Judge service, problem definitions, hidden cases, checkers |

The standalone offline-inference orchestrator lives at
`/home/qixinx/Frontier-CS/qwen_eval/frontiercs_ttt/orchestrator.py`. It shares
prompt, judge, and trace utilities but is not the training entrypoint.

### 4.3 Runtime call chain

```text
model-specific shell wrapper
  -> canonical complete-episode shell launcher
  -> Ray head/readiness validation
  -> ray job submit
  -> /home/qixinx/miles/train.py
  -> Miles rollout manager
  -> frontiercs_episode_rollout.generate_episode for each JSONL group
  -> SGLang /generate for solution and memory responses
  -> FrontierAlgorithmJudge for C++ evaluation
  -> optional external LLM memory-delta judge
  -> frontiercs_advantage.reward_post_process
  -> frontiercs_metrics.log_rollout_data
  -> Megatron actor train, checkpoint, and rollout-weight update
```

In `/home/qixinx/miles/train.py`, the order for every rollout step is strictly:

1. `rollout_manager.generate` returns complete episodes.
2. The actor trains on the returned samples.
3. A checkpoint is saved when scheduled.
4. The new actor weights are published to SGLang.

Therefore no generated sample in an episode can observe a weight update from
the middle of that episode.

## 5. Solution generation in detail

### 5.1 Prompt input

The builder is `build_act_prompt` in
`/home/qixinx/Frontier-CS/qwen_eval/frontiercs_ttt/prompts.py`.

The default prompt contains:

- a concise request to solve the optimization problem;
- focused reasoning before code;
- the current shared memory, if non-empty;
- the current problem statement;
- a warning that only the final fenced C++ block is evaluated;
- a suggestion to emit concise structured measurements to `stderr`.

It does not contain a textual problem ID. It does not contain raw earlier
diagnostics. With `FRONTIERCS_ACT_CODE_CONTEXT=none`, it does not contain the
best previous program either.

The prompt profile defaults to `qwen35_current`. Training currently does not
pass `prompt_version` into `build_act_prompt` or `build_write_prompt`; changing
the global default in the Frontier-CS registry changes training semantics.
Always use a new run ID after any prompt edit.

### 5.2 Thinking and visible answer

When `FRONTIERCS_ENABLE_THINKING=1`, the tokenizer chat template is called with
`enable_thinking=True` when supported.

The response splitter behaves as follows:

- If `</think>` is present, text before it is stored as `reasoning.txt`; text
  after it is the visible answer.
- If `<think>` is opened but never closed, the visible answer is empty. This is
  normally a length-capped invalid submission.
- If no thinking tags exist, the full response is visible.

Reasoning is retained in the trace and in the response tokens used for policy
training, but reasoning is deliberately excluded from the memory-writer prompt.

### 5.3 C++ extraction

`extract_cpp` finds all complete fences tagged `cpp` or `c++` in the visible
answer and selects the **last** one. This behavior addresses responses that
show intermediate code before a final replacement.

- Last closed `cpp`/`c++` fence: selected.
- An unclosed final tagged C++ fence: its remaining content is selected.
- A fenced answer with no explicit C++ tag: rejected.
- No fences at all: the visible response is treated as raw C++.
- Empty visible code: marked `invalid_submission`, score `0`, and never sent to
  the HTTP judge.

The current prompt explicitly says that the last block must be complete,
self-contained, compilable C++17 including headers and `main()`.

### 5.4 Parallelism and sampling

Within one round, all `G*K` calls are placed in `asyncio.gather`. Memory writing
waits until all candidates and judge feedback have returned.

The effective generation cap is:

```text
min(configured max_new_tokens, seq_length - encoded_prompt_tokens)
```

If the prompt itself occupies the whole sequence, rollout raises an error.
Sampling parameters are inherited from Miles, with an explicit deterministic
seed. The default temperature is `1.0`.

The seed includes the dataset visit index, round, problem position, and
candidate index. Repeated epochs receive different dataset visit indices;
retrying the exact same episode uses the same seed.

## 6. Frontier-CS evaluation and feedback

### 6.1 Service protocol

The adapter posts:

```text
POST <judge-url>/submit
multipart: pid=<problem-id>, lang=cpp, code=solution.cpp
```

It then polls `GET <judge-url>/result/<submission-id>` until `done`, `error`, or
the client-side timeout. The launcher first requires a healthy
`GET <judge-url>/health`.

`FRONTIERCS_JUDGE_TIMEOUT_SECONDS=1800` is the overall asynchronous polling
deadline. It is not the per-test-case runtime limit. Individual runtime limits
come from each Frontier-CS problem configuration and judge implementation.

### 6.2 Score used for training

The official judge returns a bounded normalized score in `[0,100]` and may also
return `scoreUnbounded`. The solution sample reward is:

```text
R_task = bounded_score_0_100 / 100
```

`score_unbounded` is made available to the memory writer as evidence but is not
used for the task reward or the primary W&B score curves.

The group score for round `t` is not a flat mean over every record when `K` can
vary. It is calculated as:

```text
per_problem_score[p] = mean_k score[t,p,k]
Q_t = mean_p per_problem_score[p]
```

### 6.3 Feedback exposed to the memory writer

For each candidate, the writer sees candidate code and a JSON projection with:

- `status`
- `score`
- `score_unbounded`
- `passed`
- `error`
- `case_feedback`, containing:
  - `case_index`
  - `status`
  - `score_ratio`
  - `score_ratio_unbounded`
  - `time_ms`
  - `memory_mib`
  - `has_diagnostics`
- aggregated `diagnostics` captured from program `stderr`
- `diagnostics_truncated`

Official answer output from program `stdout` is not inserted into the writer
payload. The large raw per-case `output` field is removed from stored judge API
results as well.

The intended program I/O convention is:

- required solution output goes to `stdout`;
- optional concise measurements go to `stderr`;
- `stderr` is evidence, not a direct reward target.

### 6.4 Known judge-label limitation

Some process timeouts or runtime crashes can surface in the trace as `Wrong
Answer` because the underlying judge passes empty output to the checker after a
non-successful run. The zero score remains correct, but the textual diagnosis
can be less informative than `Time Limit Exceeded`, `SIGFPE`, or `SIGSEGV`.

This behavior is currently left unchanged intentionally. Diagnostic timing and
progress output can still help later memory infer the effective failure. Do not
silently patch judge labels in one experiment: that changes the information
channel and must be treated as a separate ablation.

## 7. Memory generation in detail

### 7.1 Writer input

The writer receives exactly:

1. the previous shared memory;
2. the statement of each problem in the current group;
3. every current candidate's extracted code;
4. every current candidate's projected evaluator feedback.

It does not receive the candidate's private reasoning. It also does not receive
an out-of-band summary written by the scaffold.

The writer prompt tells the model to reason across algorithms, scores, failures,
instance observations, and transferable ideas. The visible response must use:

```text
### Cross-Problem Knowledge
### Problem-Specific Verified Findings
```

These headings are prompt instructions, not a strict parser schema. The actual
memory cleaner only removes a single outer `text` or `bytes` fence and strips
whitespace.

### 7.2 Incremental versus replacement semantics

The model is asked to update the memory, preserving, adding, revising, or
removing content according to evidence. It is not conceptually required to
discard all prior knowledge. Operationally, however, the visible writer output
is the next complete memory state.

There is deliberately no fallback:

```text
M_(t+1) = clean_memory(writer_visible_output)
```

If the writer returns empty output, `M_(t+1)` is empty. Replacing it with `M_t`
would create an off-policy mismatch: the sampled writer action would differ
from the state actually used downstream, and the delayed reward would no longer
correspond to the sampled output.

### 7.3 Input size controls

`FRONTIERCS_DIAGNOSTICS_CHARS=12000` limits the aggregate diagnostics stored by
the judge adapter for each candidate.

`FRONTIERCS_WRITER_MAX_PROMPT_CHARS=120000` is a hard pre-tokenization safety
check on the complete writer prompt. If exceeded, rollout raises rather than
silently deleting arbitrary evidence. To recover, reduce `G`, `K`, code length,
or the diagnostics limit, then use a new run ID.

## 8. Samples returned for optimization

For general `G`, `K`, and `S`, one complete episode returns:

```text
number of solution samples = G * K * S
number of memory samples   = S - 1
total trainable samples    = G*K*S + S - 1
```

When `FRONTIERCS_TRAIN_WRITE=0`:

```text
total trainable samples    = G*K*S
```

At the default `G=3`, `K=1`, `S=4`:

```text
12 solution samples + 3 memory samples = 15 trainable samples per episode
```

If exploration is enabled, there are 16 model generations per episode:
12 solution generations plus 4 memory generations. The fourth/terminal memory
generation is not packed as a training sample, so the optimizer still receives
15 samples.

Every packed sample contains:

- tokenized prompt plus generated response tokens;
- response length;
- a response-only loss mask of ones;
- rollout token log probabilities;
- SGLang weight version when available;
- completion/truncation/abort status;
- phase-specific metadata;
- a scalar raw reward.

The solution response includes the model's reasoning and visible answer, so both
are under the policy loss mask. The memory response similarly includes its
reasoning and visible memory. Only the writer **input** excludes solution
reasoning.

### 8.1 Solution metadata

Important fields are:

- `phase=act`
- unique episode trace `group_id`
- original `group_template_id`
- `episode_index`
- `memory_round`
- `problem_id`
- `candidate_index`
- `score_0_100`
- `evaluation_status`
- `executed`
- `compile_error`
- `invalid_submission`
- `has_diagnostics`
- optional `explore_score`, `explore_dims`, and `explore_brief_reason`

The unique episode trace ID has form
`<group-template-id>.episode-<eight-digit-dataset-visit-index>`. Advantage groups
use this unique ID, so repeated visits to the same template do not mix.

### 8.2 Memory metadata

Important fields are:

- `phase=write`
- unique episode `group_id`
- original `group_template_id`
- `episode_index`
- `produced_round`
- `memory_round`
- `memory_tokens`
- `memory_changed`
- `memory_empty`
- once credited: `downstream_round`, `previous_group_score_0_100`,
  `downstream_group_score_0_100`, and `write_reward_mode`

## 9. Reward assignment

### 9.1 Solution task reward

For every candidate:

```text
r_task[t,p,k] = bounded_score[t,p,k] / 100
```

This raw value is retained for reporting even when the training advantage is
centered or standardized.

### 9.2 Delayed memory reward

The memory produced after round `t` cannot be rewarded until round `t+1` is
evaluated. The rollout therefore holds the memory sample in
`pending_write_sample`, then attaches its reward in the next round.

Default `FRONTIERCS_WRITE_REWARD_MODE=delta`:

```text
r_write[t] = (Q_(t+1) - Q_t) / 100
```

Optional `downstream` mode:

```text
r_write[t] = Q_(t+1) / 100
```

There are only `S-1` trainable memory samples. The terminal `M_S` has no future
solution round, so it has no delayed task reward and is not optimized.

Negative delta rewards are allowed for the writer. The earlier decision to
avoid negative raw scores applies to the exploration judge's 0/1/2 rubric, not
to downstream performance regressions.

## 10. Advantage calculation

The implementation is
`/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_advantage.py`.
Miles receives raw rewards and a separate scalar advantage per sample; that
sample-level advantage is applied to its response tokens.

### 10.1 Default temporal solution advantage

Default `FRONTIERCS_ACT_ADVANTAGE_MODE=temporal_problem_relative` groups the
`S*K` solution samples for the same unique episode and problem:

```text
group key = (unique_episode_group_id, problem_id)
```

For rewards `r_1 ... r_n`, where `n=S*K`:

```text
mean = sum(r_i) / n
sample_std = sqrt(sum((r_i - mean)^2) / (n - 1))
A_task_i = (r_i - mean) / (sample_std + 1e-6)
```

This mode requires exactly `S*K` samples for each key and raises on incomplete
episodes. It always uses sample-standard-deviation normalization, matching the
standard GRPO convention selected for this experiment.

For `K=1`, this compares the four rounds of one problem rather than requiring
parallel sibling samples with an identical prompt.

### 10.2 Other solution advantage modes

| Mode | Behavior | Constraint/use |
|---|---|---|
| `raw` | `A_task = r_task` | Works with `K=1`; no baseline |
| `task_baseline` | subtract fixed per-problem baseline | Requires a JSON mapping for every encountered problem |
| `group_relative` | normalize candidates with key `(episode, round, problem)` | Requires `K>=2`; raises when a group has one sample |
| `temporal_problem_relative` | normalize one problem across all rounds/candidates | Current default; requires all `S*K` samples |

Baseline artifact values may be stored as `[0,1]` or `[0,100]`; values with
absolute magnitude above one are divided by 100 when loaded.

### 10.3 Memory advantage

Default `FRONTIERCS_WRITE_ADVANTAGE_MODE=direct`:

```text
A_write[t] = FRONTIERCS_WRITE_ADVANTAGE_SCALE * r_write[t]
```

Other modes:

- `positive_only`: clamp the memory reward below at zero before scaling.
- `center_by_round`: standardize writer rewards across samples that share
  `(produced_round, downstream_round)` in the current rollout batch.

Memory samples are never accidentally normalized together with solution
samples.

## 11. Optional memory-delta exploration reward

### 11.1 Intended role

Exploration reward should value useful information added to memory, including
empirical findings that may have been obtained by diagnostic printing. It does
not inspect or directly reward `stderr`, code, task score, or reasoning. Its only
semantic inputs are the memory before and after the round.

The user-message serialization intentionally matches the established
codebase-adaptation exploration judge:

```text
Previous memory M_(k-1):
{previous_memory}

Updated memory M_k:
{updated_memory}
```

Do not change this basic protocol format without an explicit experiment. The
Frontier-CS-specific rubric can differ, but common infrastructure conventions
should remain consistent.

This is important: do not add candidate artifacts as hidden judge inputs. The
current research formulation specifically defines exploration reward as a
function of two memories:

```text
E_t = Judge(M_t, M_(t+1))
```

### 11.2 Current rubric

The external LLM assigns an integer in `{0,1,2}` for each dimension:

1. `new_discoveries`
2. `error_correction`
3. `actionable_knowledge`
4. `high_level_abstraction`

The raw score is:

```text
E_t = (d1 + d2 + d3 + d4) / 8
```

Therefore `E_t` is in `[0,1]` and has no negative raw values. The prompt says
not to reward verbosity, paraphrase, generic advice, exact code recitation, or
copied instance details. It explicitly allows concrete measurements and failure
patterns to count as useful new information.

The current prompt is the `SYSTEM_PROMPT` constant in
`/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_exploration_judge.py`.
The active version is `v4-specific-testable`: an untested proposed action may
earn credit, but it must be specific and testable. The version and a hash of
both the system rubric and the memory-pair protocol are persisted in the episode
manifest and every exploration result. Any prompt change must use a new version
and a new run ID.

### 11.3 Attribution and normalization

The memory is produced from all candidates in one group round, so the same raw
`E_t` is attached to every one of that round's `G*K` solution samples. It is
then normalized separately for each problem across its full episode:

```text
normalization key = (unique_episode_group_id, problem_id)
members = all S*K candidates for that problem
A_explore[t,p,k] = standardize(E values of those members)
```

Candidates from different group episodes are never mixed to manufacture an
exploration comparison. The same round-level `E_t` sequence is repeated for
each problem in a group; task advantages still differ by problem.

Final solution advantage:

```text
A_solution = A_task + beta * A_explore
beta = FRONTIERCS_ACT_EXPLORE_BETA
```

At `beta=0`, no memory judge is called and no terminal memory is generated.

### 11.4 Why terminal memory exists

Without `M_S`, only rounds `0 ... S-2` produce memory deltas and the last
solution round has no exploration score. When beta is positive, the pipeline
therefore generates `M_S` after the final solution round. It is used only for:

```text
E_(S-1) = Judge(M_(S-1), M_S)
```

It is persisted in the trace, but it is not a memory training sample and has no
memory reward or loss.

### 11.5 API behavior and failure handling

Defaults:

- OpenAI-compatible `/chat/completions` endpoint
- model `gpt-5-mini`
- 60-second request timeout
- concurrency limit 64
- at most two attempts, separated by one second
- in-process content-hash cache, maximum 4096 entries

For GPT-5-like models the request uses `reasoning_effort=minimal`, JSON mode,
and `max_completion_tokens=2048`. Other models use temperature zero and
`max_tokens=512`.

Each round result is persisted as `exploration_reward.json`. A retry/resume
loads that file rather than paying for or resampling the judgment.

If any exploration result is missing, shaping for an affected temporal
problem-group is skipped in full. Task advantage is preserved. The code prints
a warning rather than normalizing an incomplete subset. A persisted result with
a different judge version or prompt hash is rejected instead of being silently
replayed. Always allocate a new `FRONTIERCS_RUN_ID` for every semantic change.

## 12. Full-parameter optimization

Both solution and memory samples update the same base model. There is no LoRA
adapter in the current formulation, and there is no attempt to prevent memory
gradients from changing solution behavior. This is intentional, following the
earlier successful full-parameter co-training setup.

The canonical launcher selects:

- Megatron training backend;
- Adam optimizer;
- default learning rate `1e-6`;
- constant learning-rate schedule;
- weight decay `0.1`;
- Adam betas `(0.9, 0.98)`;
- CPU optimizer offload and overlapped transfer;
- precision-aware optimizer;
- rollout log probabilities;
- GRPO advantage estimator with the custom postprocessor;
- PPO clipping `0.2` lower and `0.28` upper;
- entropy coefficient `0`;
- full activation recomputation, uniform, one layer unit;
- Flash Attention;
- attention and hidden dropout `0`;
- colocated actor and rollout GPUs.

### 12.1 KL control

Default:

```text
FRONTIERCS_KL_MODE=loss
FRONTIERCS_KL_COEF=0.01
FRONTIERCS_KL_TYPE=k1
FRONTIERCS_USE_UNBIASED_KL=0
FRONTIERCS_REF_UPDATE_INTERVAL=empty  # frozen reference
```

Supported KL estimators are `k1`, `k2`, `k3`, and `low_var_kl`.

`FRONTIERCS_KL_MODE=none` disables the KL loss. `reward` mode is explicitly
rejected because it is not compatible with the current custom-advantage path.
If `FRONTIERCS_REF_UPDATE_INTERVAL` is a positive integer, the reference is
periodically updated; empty means frozen.

### 12.2 Update size and epoch definition

`rollout_batch_size` is `B`, the number of complete group episodes per update.
`n_samples_per_prompt` must stay `1` because `K` is generated inside an episode.

For `N=30`, `B=2`:

```text
updates_per_epoch = ceil(30 / 2) = 15
```

`FRONTIERCS_NUM_EPOCHS=1` therefore defaults to 15 optimizer steps.
`FRONTIERCS_NUM_UPDATES` can override the derived total.

The launcher enables Miles dynamic global batch size. If an update returns
`B * (G*K*S + S - 1)` samples, Miles sets the global batch size to that sample
count rounded down to a multiple of data parallelism, creating one optimizer
step. The launcher pre-validates divisibility so no complete-episode samples
should be trimmed.

Required invariant:

```text
total_actor_gpus = actor_nodes * actor_gpus_per_node
model_parallel   = TP * PP * CP
DP               = total_actor_gpus / model_parallel
samples/update   = B * (G*K*S + TRAIN_WRITE*(S - 1))

total_actor_gpus must be divisible by model_parallel
samples/update must be >= DP and divisible by DP
```

For the default episode, `samples/episode=15` and `samples/update=30`.

`FRONTIERCS_NOMINAL_GLOBAL_BATCH_SIZE` is only the parser-level nominal value;
the dynamic value derived from returned samples controls the actual single
training step.

## 13. Parameter reference

Unless stated otherwise, these defaults come from the canonical complete-
episode Qwen3.5 launcher and custom config.

### 13.1 Core data and episode parameters

| Variable | Default | Meaning / constraint |
|---|---:|---|
| `FRONTIERCS_PROMPT_DATA` | canonical 30-group JSONL | Dataset path |
| `FRONTIERCS_GROUP_SIZE` | `3` | Must exactly match each row's problem count |
| `FRONTIERCS_GROUPS_PER_UPDATE` | `2` | Complete episodes per optimizer update |
| `FRONTIERCS_MEMORY_ROUNDS` | `4` | `S`; complete-episode path requires at least 2 |
| `FRONTIERCS_CANDIDATES_PER_PROBLEM` | `1` | `K`; must be at least 1 |
| `FRONTIERCS_NUM_EPOCHS` | `1` | Multiplies derived steps per epoch |
| `FRONTIERCS_NUM_UPDATES` | derived | Explicit optimizer-step override |
| `FRONTIERCS_RUN_ID` | timestamp | Unique trace/checkpoint experiment identity |
| `FRONTIERCS_TRAIN_WRITE` | `1` | `1`: co-train solution and nonterminal memory samples; `0`: train solution samples only while still generating/using memory |

### 13.2 Prompt and generation parameters

| Variable | Default | Meaning |
|---|---:|---|
| `FRONTIERCS_ACT_MAX_NEW_TOKENS` | `25600` | Maximum generated tokens for one solution response |
| `FRONTIERCS_WRITE_MAX_NEW_TOKENS` | `25600` | Maximum generated tokens for one memory response |
| `FRONTIERCS_SEQ_LENGTH` | `32768` | Maximum prompt-plus-response training sequence |
| `FRONTIERCS_MAX_TOKENS_PER_GPU` | sequence length | Megatron dynamic batching token budget per GPU |
| `FRONTIERCS_WRITER_MAX_PROMPT_CHARS` | `120000` | Hard character guard before memory generation |
| `FRONTIERCS_DIAGNOSTICS_CHARS` | `12000` | Per-candidate aggregate `stderr` retained by judge adapter |
| `FRONTIERCS_TEMPERATURE` | `1.0` | Rollout sampling temperature |
| `FRONTIERCS_ENABLE_THINKING` | `1` | Request thinking-enabled chat template |
| `FRONTIERCS_ACT_CODE_CONTEXT` | `none` | `none` or optional `best` previous-code context |
| `FRONTIERCS_LEGACY_ZERO_BEST` | unset | If `1`, allows zero-score code to be called “best”; do not use normally |

The effective solution or memory response length can be shorter than its
configured maximum because it must fit inside `FRONTIERCS_SEQ_LENGTH` after the
prompt.

### 13.3 Reward and advantage parameters

| Variable | Default | Choices / effect |
|---|---:|---|
| `FRONTIERCS_ACT_ADVANTAGE_MODE` | `temporal_problem_relative` | `raw`, `task_baseline`, `group_relative`, `temporal_problem_relative` |
| `FRONTIERCS_TASK_BASELINE_ARTIFACT` | empty | Per-problem JSON mapping used only by `task_baseline` |
| `FRONTIERCS_WRITE_REWARD_MODE` | `delta` | `delta` or `downstream` |
| `FRONTIERCS_WRITE_ADVANTAGE_MODE` | `direct` | `direct`, `positive_only`, `center_by_round` |
| `FRONTIERCS_WRITE_ADVANTAGE_SCALE` | `1.0` | Scalar multiplier after writer advantage calculation |
| `FRONTIERCS_ACT_EXPLORE_BETA` | `0` | Non-negative exploration-advantage coefficient; positive enables API judging and terminal memory |

### 13.4 External exploration judge parameters

| Variable | Default | Meaning |
|---|---|---|
| `FRONTIERCS_EXPLORE_JUDGE_API_KEY` | fallback to `OPENAI_API_KEY` | Secret API key; required when beta is positive |
| `FRONTIERCS_EXPLORE_JUDGE_API_BASE` | `https://api.openai.com/v1` | OpenAI-compatible API base |
| `FRONTIERCS_EXPLORE_JUDGE_MODEL` | `gpt-5-mini` | Memory-delta evaluator model |
| `FRONTIERCS_EXPLORE_JUDGE_TIMEOUT` | `60` | Per HTTP call timeout in seconds |
| `FRONTIERCS_EXPLORE_JUDGE_CONCURRENCY` | `64` | Process-local maximum simultaneous calls |

The concurrency limit matters across simultaneous episode coroutines. Within
one episode, each round's memory judgment is awaited before the next round.

### 13.5 Task judge parameters

| Variable | Default | Meaning |
|---|---|---|
| `FRONTIERCS_JUDGE_URL` | `http://127.0.0.1:8081` | HTTP endpoint reachable by rollout owner |
| `FRONTIERCS_JUDGE_TIMEOUT_SECONDS` | `1800` | Overall submission polling deadline |
| `FRONTIERCS_JUDGE_POLL_SECONDS` | `1` | Result polling interval |
| `FRONTIERCS_AUTO_START_JUDGE` | `0` | Start host judge from launcher when endpoint is unhealthy |
| `FRONTIERCS_JUDGE_PORT` | `8081` | Local API port for auto-start mode |
| `FRONTIERCS_GOJUDGE_PORT` | `5050` | Local go-judge port |
| `FRONTIERCS_GOJUDGE_BIN` | system path | Optional explicit go-judge executable |
| `FRONTIERCS_GOJUDGE_INIT` | system path | Optional explicit go-judge-init executable |
| `FRONTIERCS_NODE_BIN` | system path | Optional explicit Node executable |
| `FRONTIERCS_NODE_MODULES` | standard checkout location | Optional explicit Node dependency path |
| `FRONTIERCS_JUDGE_STATE_BASE` | node-local temporary storage | Optional service state root |

### 13.6 Optimizer and KL parameters

| Variable | Default | Meaning |
|---|---:|---|
| `FRONTIERCS_LR` | `1e-6` | Full-parameter actor learning rate |
| `FRONTIERCS_KL_MODE` | `loss` | `loss` or `none` |
| `FRONTIERCS_KL_COEF` | `0.01` | KL-loss coefficient |
| `FRONTIERCS_KL` | `0.01` fallback | Backward-compatible coefficient alias |
| `FRONTIERCS_KL_TYPE` | `k1` | `k1`, `k2`, `k3`, `low_var_kl` |
| `FRONTIERCS_USE_UNBIASED_KL` | `0` | Add Miles unbiased-KL option when set to `1` |
| `FRONTIERCS_REF_UPDATE_INTERVAL` | empty | Positive update interval or frozen reference |
| `FRONTIERCS_SAVE_INTERVAL` | `1` | Checkpoint frequency in rollout steps |
| `FRONTIERCS_LOGP_CHUNK` | `512` | Rollout/reference log-prob chunk size |
| `FRONTIERCS_DIST_TIMEOUT_MIN` | `120` | Distributed operation timeout in minutes |

### 13.7 Paths, model, and W&B parameters

| Variable | Default | Meaning |
|---|---|---|
| `FRONTIERCS_ROOT` | sibling Frontier-CS discovery | Required benchmark checkout |
| `FRONTIERCS_HF_CHECKPOINT` | none | Required Hugging Face model directory for SGLang |
| `FRONTIERCS_TORCH_DIST` | none | Required converted Megatron checkpoint |
| `FRONTIERCS_MODEL_CONFIG_SCRIPT` | Qwen3.5-4B model config | Miles model-architecture shell fragment |
| `FRONTIERCS_MODEL_LABEL` | `qwen3.5-4B` | Run label only |
| `FRONTIERCS_OUTPUT_ROOT` | Frontier-CS result root | Parent for traces and run state |
| `FRONTIERCS_SAVE_DIR` | run-local checkpoints | Megatron checkpoint directory |
| `FRONTIERCS_LOG_DIR` | run-local logs | Launcher/job log directory |
| `FRONTIERCS_USE_WANDB` | `1` | Disabled automatically without key unless offline mode |
| `FRONTIERCS_WANDB_ENV_FILE` | empty | Private shell env containing W&B credentials |
| `FRONTIERCS_WANDB_RUN_ID` | empty | Optional stable W&B run ID |
| `FRONTIERCS_WANDB_DIR` | empty | Optional W&B local storage directory |
| `WANDB_PROJECT` | `miles-frontier-cs` | W&B project |
| `WANDB_GROUP` | model label plus run ID | W&B grouping |
| `WANDB_TEAM` | empty | Optional W&B entity/team |
| `WANDB_MODE` | online default | Can be `offline` without a key |

Never put API keys in a committed env file, Ray command line, trace, or report.
The launcher forwards required secrets through the Ray runtime environment.

### 13.8 Hardware and Ray parameters

| Variable | Default | Meaning |
|---|---:|---|
| `FRONTIERCS_NGPU` | visible GPU count | GPUs visible on the local node; normally allocate the full node |
| `FRONTIERCS_TP` | local visible GPU count | Tensor parallelism; Qwen3.6 wrapper defaults to `4` |
| `FRONTIERCS_PP` | `1` | Pipeline parallelism |
| `FRONTIERCS_CP` | `1` | Context parallelism |
| `FRONTIERCS_ACTOR_NUM_NODES` | `1` | Number of actor/Ray GPU nodes |
| `FRONTIERCS_ACTOR_GPUS_PER_NODE` | local GPU count | Actor GPUs consumed on each node |
| `FRONTIERCS_RAY_NODE_GPUS` | actor GPUs per node | GPUs advertised by each Ray node |
| `FRONTIERCS_ROLLOUT_NUM_GPUS` | total actor GPUs | Total rollout placement GPUs |
| `FRONTIERCS_ROLLOUT_GPUS_PER_ENGINE` | `1` | GPUs assigned to each SGLang rollout engine |
| `FRONTIERCS_SGLANG_MEM_FRACTION` | `0.5` | Static SGLang memory fraction |
| `FRONTIERCS_START_RAY` | `1` | Start a head locally; set `0` for externally managed Ray |
| `FRONTIERCS_RESET_RAY` | `0` | Stop a stale local Ray before launch |
| `FRONTIERCS_STOP_RAY_ON_EXIT` | `0`, multi-node head defaults `1` | Stop head service at launcher exit |
| `FRONTIERCS_RAY_ADDRESS` | local dashboard URL | Ray Jobs API endpoint |
| `FRONTIERCS_RAY_CLUSTER_ADDRESS` | derived GCS address | Core Ray client address for readiness check |
| `FRONTIERCS_WAIT_FOR_RAY_CLUSTER` | `1` | Require expected resources before submit |
| `FRONTIERCS_CHECK_RAY_PATHS` | `1` | Check required shared paths on every eligible node |
| `FRONTIERCS_RAY_WAIT_TIMEOUT_SECONDS` | `600` | Resource-readiness timeout |
| `FRONTIERCS_RAY_WAIT_POLL_SECONDS` | `5` | Readiness polling interval |
| `FRONTIERCS_RAY_CONNECT_TIMEOUT_SECONDS` | `600` | Worker wait for head |
| `FRONTIERCS_RAY_WORKER_MONITOR_SECONDS` | `5` | Worker head-health interval |
| `FRONTIERCS_RAY_HEAD_MISSING_CHECKS` | `6` | Failed checks before worker exits |
| `RAY_GCS_PORT` | `6379` | Ray GCS port |
| `RAY_DASH_PORT` | `8265` | Ray dashboard/jobs port |
| `RAY_OBJECT_STORE_MEMORY` | Ray default | Optional bytes per node |
| `MASTER_ADDR` | `127.0.0.1` | Head IP for a local run; explicit in multi-node runs |

## 14. Single-node and multi-node launch

### 14.1 Shared prerequisites

Every GPU node must see the same absolute paths for:

- Miles checkout;
- Frontier-CS checkout;
- group JSONL;
- Hugging Face checkpoint;
- converted Megatron checkpoint;
- trace output root;
- checkpoint output root.

The launcher verifies these through Ray by default. Do not disable the check to
paper over a missing shared mount.

The judge endpoint must be reachable by the rollout owner. Run the independent
Docker judge from `/home/qixinx/Frontier-CS/algorithmic` when privileged Docker
is available, or use the host-judge wrapper only after validating go-judge and
cgroup support.

### 14.2 Single-node launch skeleton

Use a private environment file based on
`/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_env.example`, then:

```bash
source /home/<user>/frontiercs.env
cd /home/<user>/miles
bash /home/<user>/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.5_4B.sh
```

For Qwen3.6-27B:

```bash
source /home/<user>/frontiercs.env
bash /home/<user>/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.6_27B.sh
```

The Qwen3.6 wrapper only changes the model config, model label, and default TP.
All pipeline, reward, and optimizer settings remain controlled by the same
canonical launcher.

### 14.3 Multi-node behavior

On all nodes, set the same node count, GPUs per node, model/data paths, run ID,
and reward parameters. Call the model-specific wrapper with `head` on one node
and `worker` on every other node.

Example topology only—not a universally optimal model configuration:

```bash
export FRONTIERCS_ACTOR_NUM_NODES=4
export FRONTIERCS_ACTOR_GPUS_PER_NODE=8
export FRONTIERCS_NGPU=8
export FRONTIERCS_TP=8
export FRONTIERCS_PP=2
export FRONTIERCS_CP=1
```

Head:

```bash
bash /home/<user>/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.6_27B.sh head <head-ip>
```

Workers:

```bash
bash /home/<user>/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.6_27B.sh worker <head-ip>
```

The head starts Ray, waits for the full declared cluster, checks shared paths,
and is the only process that submits training. Workers join Ray, monitor the
head, and stop their local Ray services after the head disappears.

The rollout manager is pinned to the head. Actor and rollout resources are
colocated. Multi-node compatibility does not imply that arbitrary `TP/PP/CP`
settings will fit a particular model; calculate model memory and the DP/sample
divisibility invariant before launch.

## 15. Trace format, replay, and recovery

The default current-machine trace parent is
`/home/qixinx/Frontier-CS/qwen_eval/results/frontiercs_ttt_rl`.
Each run lives under `<output-root>/<run-id>`.

Default episode tree:

```text
/home/qixinx/Frontier-CS/qwen_eval/results/frontiercs_ttt_rl/<run-id>/
  groups/
    <template>.episode-00000007/
      episode_manifest.json
      episode_state.json
      episode.json
      round_000/
        memory_in.md
        write_prompt.txt
        write_output.txt
        write_reasoning.txt
        memory_out.md
        exploration_reward.json       # only when beta > 0
        round.json
        problems/
          <problem-id>/
            candidate_00/
              act_prompt.txt
              response.txt
              reasoning.txt
              solution.cpp
              feedback.json
              record.json
              train_sample.json
```

Meanings:

- `response.txt`: complete raw model response, including thinking tags.
- `reasoning.txt`: extracted private thinking portion.
- `solution.cpp`: final extracted visible C++ submission.
- `feedback.json`: only the feedback projection visible to the writer.
- `record.json`: full serialized candidate record.
- `train_sample.json`: exact packed Miles sample.
- `memory_in.md` / `memory_out.md`: state transition for the round.
- `write_*`: raw writer prompt, response, and extracted reasoning.
- `exploration_reward.json`: persisted result or persisted unavailable status.
- `round.json`: atomic round commit including samples added and state after.
- `episode_state.json`: resumable working state.
- `episode.json`: final episode commit and exact returned sample list.

Candidate records and packed samples are reused if already complete. A complete
`episode.json` returns its stored sample list without new generation, judging,
or API calls.

The manifest rejects changes to several core episode settings. It is not a
complete hash of all prompts and external services. The safe operational rule
is stronger:

> Never reuse a run ID after changing any model, checkpoint, prompt, dataset,
> judge behavior, reward rule, advantage rule, generation limit, or beta.

Atomic JSON/text writes use a temporary file followed by `os.replace`, so a
process crash should leave either the old complete file or the new complete
file rather than a partially written commit.

## 16. W&B metrics

The default project is `miles-frontier-cs`. Metrics intentionally omit the
redundant `frontiercs/` prefix. No prompt, code, memory text, diagnostics text,
or case viewer is uploaded.

### Score and execution by round

For every `r` in `0 ... S-1`:

- `score/current_mean_r<r>`: mean bounded score of candidates generated in
  that round.
- `score/positive_frac_r<r>`: fraction of candidates with score above zero.
- `score/best_mean_r<r>`: mean per episode/problem cumulative-best score through
  that round.
- `act/executed_frac_r<r>`: fraction whose feedback status indicates execution.

`current_mean` is the score of the current round only. `best_mean` is monotonic
by construction because it uses the best score seen through that round.

### Aggregate failure and length metrics

- `act/executed_frac`
- `act/compile_error_frac`
- `act/invalid_submission_frac`
- `act/length_stop_frac`
- `write/length_stop_frac`
- `sample_length/act_mean`
- `sample_length/write_mean`
- `diagnostics/nonempty_frac`

`compile_error` is currently detected by matching `compile failed` in the
feedback error string. `invalid_submission` means no non-empty visible C++ was
extracted. These categories need not cover every zero-score outcome.

### Memory metrics

- `memory/changed_frac`
- `memory/empty_frac`
- `memory_length/after_r0_mean` through
  `memory_length/after_r(S-2)_mean`

Terminal memory length is not included because terminal memory is not a
trainable writer sample.

### Training-signal metrics

- `training_signal/write_reward_mean`
- `training_signal/grpo_zero_std_group_frac`
- `training_signal/act_advantage_abs_mean`
- `training_signal/write_advantage_abs_mean`

### Exploration-reward metrics

- `exploration_reward/new_discoveries_mean`
- `exploration_reward/error_correction_mean`
- `exploration_reward/actionable_knowledge_mean`
- `exploration_reward/high_level_abstraction_mean`
- `exploration_reward/reward_mean`
- `exploration_reward/group_zero_std_frac`
- `exploration_reward/group_std_mean`

Interpretation:

- The four dimension means and `reward_mean` are computed once per unique
  episode/round judgment, rather than overcounting the copies attached to all
  `G*K` ACT samples.
- Exploration group statistics use the exact normalization partition
  `(unique_episode_group_id, problem_id)` with all `S*K` candidates. A high
  `group_zero_std_frac` means the judge gives no relative signal across rounds
  for many episode/problem groups. `group_std_mean` is the mean sample standard
  deviation used as the GRPO denominator before epsilon is added.
- A high task GRPO zero-std fraction means all rounds achieved the same task
  reward for many problems, often all zero.
- Near-zero mean absolute writer advantage means memories are not receiving a
  useful delayed performance signal even if they are non-empty.

## 17. Validation already completed

The current Frontier-CS unit suite covers:

- complete episode sample counts;
- frozen weight-version propagation;
- delayed writer reward;
- no diagnostics in later solution prompts;
- diagnostics and code in writer prompts;
- exclusion of solution reasoning from writer prompts;
- exact disk replay without repeated inference/judging;
- no memory fallback;
- terminal memory generation and non-training status;
- exploration score attachment and normalization;
- missing exploration score fail-closed behavior;
- task and writer advantage modes;
- metrics;
- host judge utilities;
- Ray-cluster readiness checks.

The last local result was 22 passing tests. Run again after every change:

```bash
PYTHONPATH=/home/qixinx/miles:/home/qixinx/Frontier-CS \
  /home/qixinx/Frontier-CS/.venv/bin/pytest -q -p no:cacheprovider \
  /home/qixinx/miles/examples/frontiercs_ttt/tests
```

Also run Ruff on every changed Frontier-CS Python file using the active
environment's Ruff executable, and shell syntax checks:

```bash
bash -n /home/qixinx/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.5_4B.sh
bash -n /home/qixinx/miles/examples/frontiercs_ttt/run_frontiercs_ttt_episode_qwen3.6_27B.sh
bash -n /home/qixinx/miles/examples/frontiercs_ttt/run_frontiercs_ttt_multinode.sh
```

Mocked tests do not validate real SGLang responses, a live Frontier-CS judge,
external API credentials, Ray resource placement, Megatron memory consumption,
or a real optimizer update.

## 18. Required autonomous experiment procedure

An agent taking over experiments should follow these checkpoints in order.

### Checkpoint A: immutable experiment declaration

Before touching GPUs, write down:

- Miles commit;
- Frontier-CS commit;
- model checkpoint and conversion checkpoint;
- dataset file and row count;
- `G`, `K`, `S`, and `B`;
- sequence and generation lengths;
- task advantage mode;
- writer reward/advantage modes and scale;
- beta, exploration judge model, and exploration prompt hash/content;
- KL settings, learning rate, topology, number of updates;
- fresh run ID, trace root, checkpoint root, W&B run ID.

Do not continue if any semantic setting is implicit or shares an old run ID.

### Checkpoint B: CPU/unit preflight

1. Inspect working trees and preserve unrelated changes.
2. Run all Frontier-CS tests.
3. Run Ruff and shell syntax checks.
4. Validate the JSONL line count and each row's problem count.
5. Confirm all problem statement directories exist.
6. Confirm judge health and submit one known compilable test program.
7. If beta is positive, call the memory judge on two synthetic pairs and check
   valid JSON, all four integer dimensions, score range, and trace persistence.

### Checkpoint C: one-group rollout-only semantic smoke

Use the smoke JSONL and a fresh run ID. Before a full training job, verify from
the trace that:

- there are exactly `S` solution rounds;
- every round has `G*K` candidate directories;
- round 1's solution prompt contains `M1` but no raw round-0 diagnostics;
- writer prompt contains code and feedback but no solution reasoning;
- every nonterminal writer output becomes the next round's exact memory;
- empty writer output would remain empty rather than fallback;
- with beta positive, there are `S` exploration JSON files and a terminal
  memory after the final round;
- only `S-1` memory train samples exist.

### Checkpoint D: one real optimizer step

Run one update with the intended model and topology. Success requires:

- Ray reports every requested node and GPU;
- path validation passes on every GPU node;
- launcher reports expected DP and exact samples per update;
- no sample trimming appears in logs;
- the actor completes one optimizer step;
- a checkpoint is written and can be enumerated;
- rollout weights update after training;
- W&B has non-empty round score and advantage metrics;
- trace `episode.json` sample counts match the formula;
- with beta positive, exploration coverage is exactly `1.0`.

### Checkpoint E: resume test

Interrupt only after an atomic round or episode commit, then resume with the
same configuration and run ID. Confirm completed candidates/episodes are loaded
from disk and not resampled or re-judged. Then use a new run ID for the real
experiment.

### Checkpoint F: full run monitoring

Monitor at least:

- Ray job state and node liveness;
- judge health and pending submission growth;
- current and cumulative-best scores per round;
- invalid, compile, execution, and length-stop fractions;
- memory empty/change fractions and lengths;
- task and writer advantage magnitude;
- exploration coverage, variance, and API errors;
- checkpoint creation and filesystem capacity.

Stop and diagnose rather than consuming a full allocation when any hard-stop
condition below occurs.

## 19. Hard-stop conditions

Stop a run when:

- exploration beta is positive but coverage is below `1.0` for a completed
  update;
- the trace lacks the terminal memory or final-round exploration file;
- episode sample count differs from `G*K*S + S - 1`;
- Miles trims samples despite launcher validation;
- task or exploration advantages contain NaN/Inf;
- Ray has fewer nodes/GPUs than requested;
- required shared paths differ across nodes;
- judge health fails persistently or all evaluations return transport errors;
- W&B score metrics are blank after a completed rollout;
- a reused run ID reports manifest or cached artifacts from different
  semantics;
- memory emptiness or generation length stops spike enough to invalidate the
  intended mechanism;
- disk capacity threatens trace/checkpoint atomic writes.

Zero task reward by itself is not an infrastructure stop condition. It can be a
real model or algorithm failure and should be analyzed through code, compile
feedback, runtime behavior, and diagnostics.

## 20. Recommended first exploration-reward experiments

Do not begin with a large beta sweep. First establish whether the judge signal
is meaningful.

### Phase 1: offline judge calibration

Sample real consecutive memory pairs from existing traces and manually label:

- no substantive change;
- verbose paraphrase;
- new but narrow empirical fact;
- corrected false belief;
- concrete new search/implementation strategy;
- transferable high-level abstraction;
- plausible but unsupported fabricated claim;
- memory deletion or regression.

Run the external judge repeatedly and inspect agreement, stability, verbosity
bias, and whether false specificity is over-rewarded. Revise and version the
prompt before training.

### Phase 2: exact controlled comparison

Run at least:

1. beta `0`, canonical pipeline;
2. one small positive beta, all other settings and seeds held fixed as far as
   the API path permits.

Compare task-score trajectories, exploration coverage/variance, memory content,
writer reward, and failure rates. Do not attribute gains to exploration reward
if prompt, model, group data, context length, or judge behavior also changed.

### Phase 3: beta sweep only after signal validation

The exploration advantage is standardized before beta is applied, so beta is
an advantage-scale coefficient, not a raw `[0,1]` reward weight. Reasonable
candidate values must be chosen relative to observed
`act_advantage_abs_mean`, not from the raw rubric range alone.

## 21. Known risks and design tradeoffs

### 21.1 Coarse exploration credit

One group memory is generated from all `G*K` candidates, so all candidates in a
round receive the same exploration score. This is the unavoidable current
credit granularity. Candidate-level exploration would require a different
memory/action formulation.

### 21.2 Judge cannot verify truth

Because the external judge sees only two memories, it can judge apparent
information gain but cannot verify claims against code, diagnostics, or task
ground truth. A model might learn to add plausible detailed claims. Task reward,
manual calibration, prompt anti-hacking language, and correlation analysis are
needed to constrain this.

### 21.3 Repeated problems and memorization

Repeated membership is intentional for sharing knowledge, but a small set can
encourage exact-solution memorization instead of meta-level memory. Inspect
memory for copied programs, instance-specific answer recitation, and problem-ID
lookup tables. The writer prompt discourages exact code, but there is no hard
filter.

### 21.4 Long-generation failures

Earlier offline Qwen runs showed unclosed thinking, response-length stops,
invalid code extraction, and compilation failures. Increasing context does not
guarantee a valid final program. Track solution/memory length stops separately
and inspect `response.txt`, `reasoning.txt`, and `solution.cpp` together.

### 21.5 Writer context pressure

Writer input grows with `G`, `K`, code length, problem statement length, and
diagnostics. `120000` characters is a guard, not a guarantee that the tokenized
prompt plus a `25600`-token response fits the `32768` sequence. Effective output
length is automatically reduced by prompt tokens.

### 21.6 External API cost and latency

At beta positive, one complete episode makes `S` logical memory judgments. For
30 groups and one epoch, that is 120 successful judgments before retries. Calls
from different episodes can overlap, but each episode waits for its current
judgment before proceeding. Persisted results prevent duplicate cost on resume.

### 21.7 Terminal memory is not learned directly

The final memory is generated solely to score final-round solution exploration.
It is not a writer sample because no downstream task round exists to define its
writer reward. Changing this requires a new explicit terminal objective, not
silently adding it to training.

### 21.8 Best-code context is an ablation

`FRONTIERCS_ACT_CODE_CONTEXT=best` can expose the highest positive-scoring prior
program. Zero-score code is omitted by default and the prompt says no positive
solution exists. The canonical formulation is `none`: cross-round information
must travel through memory. Do not enable best-code context in the main run
without labeling it as a different scaffold.

## 22. Rules for future code changes

1. Preserve the complete frozen-policy episode boundary.
2. Preserve the default absence of previous diagnostics/code in solution input.
3. Preserve solution-reasoning exclusion from writer input.
4. Preserve no-fallback memory semantics.
5. Preserve bounded score for task reward and expose unbounded score only as
   evidence unless a new experiment explicitly changes this.
6. Preserve within-episode, same-problem temporal normalization; never normalize
   exploration reward across unrelated groups.
7. Preserve the final terminal-memory distinction: judged, traced, not trained.
8. Never mutate the older round-wise path and assume the complete-episode path
   changed too; test the actual canonical entrypoint.
9. Add semantic settings and prompt hashes to manifests when extending
   versioning.
10. Add a focused unit test for every reward, trace, retry, or sample-count
    behavior change.
11. Use a fresh run ID for every semantic change.
12. Do not include unrelated dirty files in commits.
13. Never commit API keys, W&B tokens, model weights, hidden-output dumps, or
    cluster-specific credentials.

## 23. Minimum experiment report template

Every completed or failed experiment should leave a short Markdown report with:

```text
Experiment name / run ID:
Date and cluster allocation:
Miles commit:
Frontier-CS commit:
Model and checkpoint:
Dataset and group count:
G / K / S / B:
Context, solution max, memory max:
Prompt profile and prompt hash:
Task advantage mode:
Writer reward / advantage / scale:
Exploration beta / judge model / rubric version:
KL mode / coefficient / estimator:
LR / optimizer steps / topology TP-PP-CP-DP:
Judge endpoint and benchmark revision:
Trace root:
Checkpoint root:
W&B project and run ID:

Per-round current mean:
Per-round cumulative-best mean:
Positive/executed/compile/invalid/length-stop rates:
Memory empty/change/length summary:
Task/write/exploration advantage summary:
Exploration coverage and zero-std fraction:

Representative positive cases:
Representative regressions/failures:
Infrastructure incidents and retries:
Conclusion:
Next action:
```

Use direct trace evidence for case claims. Never infer that memory caused a gain
merely because a later-round score is higher; inspect the memory delta, next
solution, diagnostics, and algorithmic change.

## 24. Immediate next actions for the takeover agent

1. Inspect and commit the current Frontier-CS-only working-tree changes in
   `/home/qixinx/miles` without touching unrelated codebase-adaptation files.
2. Finish polishing and versioning the exploration judge prompt in
   `/home/qixinx/miles/examples/frontiercs_ttt/frontiercs_exploration_judge.py`.
3. Build a small labeled calibration artifact from real memory pairs and record
   judge outputs.
4. Add exploration prompt/model/hash to the episode manifest to prevent stale
   cache reuse.
5. Run the full 22-test suite, Ruff, and shell checks.
6. Run one live external-judge semantic smoke.
7. Run one real Qwen3.6-27B optimizer step on the collaborator's multi-node
   allocation with a fresh run ID.
8. Verify all hard-stop conditions before launching the full 30-group run.

This order is intentional: prompt quality and cache identity must be resolved
before spending a large multi-node allocation on exploration-reward training.
