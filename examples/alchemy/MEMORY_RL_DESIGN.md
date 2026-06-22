# Memory-RL design — training *writing* and *using* memory with GRPO rewards (Symbolic Alchemy)

Development doc for adapting our GRPO loop to co-train two abilities: **writing** a memory and **using** it to
act. **Everything here is REWARD-based and fed to GRPO — there is NO loss anywhere** (no NLL, no "L_WM loss";
we compute scalar rewards and let policy gradient do the rest). Pairs with the offline-eval findings in
`eval/BASELINE_RESULTS.md`, `eval/QWEN_SUMMARY_DRIFT.md`, the two `eval/*_DRAFT.md` case analyses, and the
validation scripts under `eval/wm_*.py`.

Status: design largely settled; the **writing-memory reward** signal is being validated offline (latest:
generation-accuracy with a REASONING/ANSWER step — see §3.4). Not yet implemented in training. Last updated 2026-06-16.

---

## 1. Why

We will **not** hand-engineer a per-task heuristic memory prompt — brittle and non-scalable (the structured
prompt is an alchemy-specific patch; `ep6` shows even a good prompt fails when the *content* written is wrong).
We want the model to *learn* (a) to write a task-appropriate memory and (b) to use it. Offline eval established
both are real, roughly-equal bottlenecks with a measurable Qwen3.5-4B < Claude gap (writer/reader 2×2:
0.319 self → 0.527 / 0.520 single-fix → 0.618 self).

The blocker: the task reward (trial score) is too sparse/distal to train memory *writing* directly, and because
the memory is **discrete sampled text**, you cannot backprop any prediction signal through it into the writer.
So we give the memory-writing action its own **dense, local reward** (how well the memory predicts the future),
and train it by policy gradient — same as the action reward. Two reward streams, one GRPO loop, no loss.

---

## 2. Episode decomposition — two reward streams

A 10-trial episode splits into two process types, each its own GRPO stream on its own token span. **Both are
rewards; group → standardize → broadcast the scalar to that span's tokens.**

| stream | token span | count | reward |
|---|---|---|---|
| **ACT** (use memory to act) | a trial's action tokens | 10 trials | the trial's normalized score (§4) |
| **WRITE** (write memory `M_k`) | the memory tokens | 9 (after trials 1–9) | the memory's prediction quality on `F_k` (§3) |

Trial 1 acts with empty memory (its observations feed `M_1`); trials 2–10 act on `M_1..M_9`. Both streams keep
**KL-to-ref** (pure GRPO collapses the writer's output syntax — see `EXPERIMENT_LOG.md`).

> Granularity note (open, §6): the current `alchemy_rollout.py` does **per-step** REWRITE; the eval harness
> summarizes **per-trial**. The WRITE reward below works at either granularity; pick one and align.

---

## 3. The WRITE reward — predict the future

**Rejected: a constructed probe set** (needs privileged chemistry; not general). The general, transferable
signal is **predicting the real future**.

**Keystone.** Within an episode the chemistry is fixed, so *any* transition observed after a REWRITE point is a
valid held-out test of whether `M_k` captured the chemistry. So we reuse the **real observed future as the test
set**, and score **all G candidate memories at the same REWRITE point against the same `F_k`** — clean GRPO group,
no rollout blow-up.

### Procedure at REWRITE point `k`
1. From the same context (trials `1..k`) sample `G` candidate memories `{M_k^(1..G)}`.
2. `F_k` = the real potion-application transitions `(s, a, d)` (`d` = resulting stone + reward) observed in
   trials `> k` of the rollout. **Include no-ops** (a good memory should predict "this potion is blocked/inert
   here → no change"); they are valid tests, and they fix the sparsity of changed-only `F_k`.
3. Score every candidate on the **same** `F_k`. The per-memory reward is its **prediction quality** on `F_k`
   (metric in §3.4). Same `F_k` for all G ⇒ any "difficulty" or no-memory baseline is a shared constant.
4. **Standardize the G rewards (GRPO) → broadcast the advantage to that candidate's memory tokens.**

This gives: generality (real future), clean GRPO grouping (one future per branch), and anti-drift (a "never
transform" memory mispredicts the real transformations → low reward → negative advantage).

### 3.1 No ∅ baseline needed
The reward is a per-memory scalar; GRPO standardizes within the REWRITE group. A no-memory (∅) baseline is the
*same constant* for all G candidates (same `F_k`, independent of `M`) ⇒ it **cancels exactly** in
`A^i = (r^i − mean_G)/std_G`. So we do **not** subtract ∅ (we don't even compute it in training); the advantage
depends only on the candidates' rewards *relative to each other*. (∅ stays a validation diagnostic only.)

### 3.2 Predictor = the policy itself
The "prediction" is produced by the **online policy θ** (detached — used only to emit the scalar reward), given
**memory + the local `(s, a)`, NOT the raw history** (else the memory has ~0 marginal value). One model, three
roles: writer / actor / predictor. Frozen-ref predictor kept as an anti-self-collusion ablation.

### 3.3 Memory length capped (replace-style) — forces compression; also avoids the long-memory retrieval burden
the weak model shows (§3.4).

### 3.4 Reward metric — validation status (offline, on served Qwen3.5-4B)
We tried two metrics for "prediction quality on `F_k`":
- **logP (teacher-forced)** `logP_θ(d | M, s, a)`: separates *good* memory (oracle/Claude) from nothing cleanly,
  BUT is **length-confounded** and does **not** rank realistic Qwen-quality memories within a group (within-group
  logP tracked memory *length*, not quality). Demoted to a **monitoring metric**, not the training signal.
- **generation accuracy** (current): the policy GENERATES `d_hat` for each `(s,a)` and we score `d_hat == d`
  (exact = colour/size/shape/reward; feature = the 3 features). More robust (measures the actual prediction,
  not a fragile per-token likelihood; harder to hack). Key finding: with a *bare-answer* prompt even the oracle
  memory floored (~0.4) because the no-reasoning model **wouldn't apply the written rule** (41% "predicted
  no-change", 100% contradicting an explicit rule in the notes). Fix = let it **reason then answer** via a
  `REASONING: … ANSWER: …` format (still `enable_thinking=False`, mirrors the actor) → oracle jumps to ~0.79.
  The full within-group separation under this reasoning-generation is the run in flight.

Open: exact vs feature match (reward number is part of the env, so exact is the principled target; feature is a
softer fallback); `F_k` pooling vs single future (sparsity, §6).

---

## 4. The ACT reward — using memory to act

**Per-trial reward = the trial's normalized score** = `trial_k_score / oracle_trial_k_score` (oracle precomputed
for all 1000 episodes). This is the actual task objective ("given the memory you have, play this trial well").

- **Grouping → standardize → broadcast**: GRPO over the `G` rollouts **per trial position** — standardize the G
  rollouts' trial-`k` scores, then **broadcast** that single scalar advantage to all of trial `k`'s action tokens
  in that rollout. (Not "averaged over tokens" — one normalized scalar per trial, broadcast.)
- **Trial 1** (empty memory, exploration): reward it by its own score too (v1; revisit only if it suppresses
  exploration).
- Within-trial credit: broadcast the trial scalar to all the trial's action tokens (return-to-go is a later
  refinement).

### The two streams stay separate (for now)
ACT tokens get the task reward; WRITE tokens get the prediction reward. **The writer does NOT receive any
downstream task reward / discounted future return for now** (deliberately deferred — keep it simple). Nice
self-consistency: the transitions the actor observes in trial `k` are simultaneously (a) the basis of trial
`k`'s ACT reward and (b) the test set `F_k` for earlier memories' WRITE reward — one set of data, two uses.

---

## 5. Rejected / non-options (so we don't revisit)
- **Probe set from known chemistry** — not general.
- **Any differentiable loss on the writer** (NLL etc.) — gradient cannot pass through discrete sampled memory;
  we use rewards only.
- **Branching the full episode per memory candidate** — `G^9` blow-up; avoided by the shared-`F_k` trick (§3).
- **Predictor that also sees raw history** — kills the memory's marginal value.
- **A ∅-baseline subtraction in the WRITE reward** — redundant; cancels in the GRPO group (§3.1).

---

## 6. Open questions / risks
1. **`F_k` sparsity.** Single-future `F_k` is often tiny (2–7 unique transitions; weak actors waste moves on
   no-ops and repeat a few moves) → the per-point reward is coarse/high-variance. Including no-ops helps; may
   still need to **pool `F_k` across the `G` group trajectories' futures**. (Single vs union: leaning union.)
2. **REWRITE granularity** per-step (current rollout) vs per-trial (eval) — pick one, align.
3. **`λ`** (ACT vs WRITE stream weight) — start 1 after per-group standardization; tune.
4. **ACT mode in training**: `augment` (history crutch, easier early) vs `replace` (memory is sole carrier,
   matches the predictor's memory-only view). Predictor is memory-only regardless.
5. **Reward hacking / self-collusion** of the same-model predictor — frozen-ref ablation tests it.
6. **Generation cost** of the accuracy reward (autoregressive `d_hat` with reasoning) vs the cheaper logP score —
   monitor; logP can stay a cheap monitoring signal.

---

## 7. Implementation sketch (fill when §6 locked)
Target: `alchemy_rollout.py` (ACT→REWRITE loop exists; `generate()` records `memory_in`, `raw_rewrite`,
`memory`, transition `d`). Work:
- Tag token spans (ACT-action vs WRITE-memory) for stream-specific advantages.
- Per REWRITE point: sample `G` memories from the same context; build `F_k` from the future records; score each
  candidate's prediction quality on `F_k`.
- Per trial: compute normalized trial score.
- Emit two advantage tensors (task reward on ACT spans, prediction reward on WRITE spans), each
  group→standardize→broadcast; combine with `λ`; keep KL-to-ref.

## 8. Eval plan
Reuse the offline harness; save base vs trained checkpoints; compare on **hard-20**:
- `performance` (normalized-to-oracle) and `I_score` under replace.
- Whether the 4B's drift is fixed **without** the structured-prompt scaffold (ep55 chain: does the +15 recipe
  survive `M_1 → M_9` under a plain prompt?).
- Stretch: evidence the closed gap is **memory-specific** (memory-only prediction accuracy on held-out
  transitions, base vs trained).
