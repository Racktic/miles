# Symbolic Alchemy — Frontier-Model Eval Baseline

Core results document for the zero-shot, no-summary Symbolic Alchemy eval (TMLR-aligned). **Append new
models/conditions here as they are run.** Pipeline & method live in `examples/alchemy/eval/`.

## Setup

- **Task**: DeepMind Symbolic Alchemy, level `perceptual_mapping_randomized_with_random_bottleneck`
  (the hardest / random-bottleneck level). 1000 prebuilt episodes (`examples/alchemy/data/...`).
- **Protocol** (TMLR, Sawyer et al. 2025): **zero-shot, no-summary** (full episode history in context every
  turn, no memory writes), **10 trials/episode, 20 steps/trial, end-trial enabled**. Response is the
  3-part `OBSERVATION / REASONING / ACTION`. System prompt is VERBATIM TMLR Figure 9; `--prior-info`
  toggles the no-prior (Fig 5a) vs prior-info (Fig 5b, +reward/potion-pairing/causal invariants) condition.
- **Oracle**: exact KNOWN-chemistry per-trial optimum ("optimal actions for the given items") computed by
  cloning the real env + exhaustive memoised search → provably optimal, ground-truth dynamics. (The
  dm_alchemy Bayesian ideal-observer is intractable here and is NOT what TMLR normalizes to.)
- **Metrics** (per trial: `normalized = agent_score / oracle_score`):
  - **performance** = mean normalized score over the 10 trials (1.0 = plays optimally).
  - **I_score (robust)** = mean(normalized trials 6-10) − mean(trials 1-5) — primary, robust to trial-1 noise.
  - **I_score (TMLR)** = mean(trials 6-10) − trial 1 — the paper's exact metric (kept for reference).
- All runs below: **0 invalid actions** (parser anchored on `^ACTION:`). N reported per row; ± is SE of the mean.

### Episode sets
- **random-20**: episodes 0-19 (mixed difficulty — includes 2 no-bottleneck episodes; reference set).
- **hard-20** (`data/hard_set_20.json`): all 7-edge (hardest, max bottleneck) episodes —
  `[1,3,6,8,9,11,15,16,19,26,29,30,37,41,51,52,53,55,59,64]`. **Use this as the main comparison set**
  (no-bottleneck episodes inflate scores and have little to meta-learn).

> **Caveat**: do NOT compare these numbers head-to-head with TMLR's figures — different episodes/items, so
> it's a loose distributional comparison at best. Within this document, models ARE comparable (same episode sets).

---

## Results

### performance (normalized-to-oracle, ↑ better; 1.0 = optimal)

| Model | Condition | Set | N | performance | I_score (robust) | I_score (TMLR) | raw/45 |
|---|---|---|---|---|---|---|---|
| claude-opus-4-8 | no-prior | random-20 | 20 | 0.709 ± 0.046 | +0.057 ± 0.033 | +0.212 ± 0.077 | 18.94 |
| claude-opus-4-8 | no-prior | **hard-20** | 20 | **0.674 ± 0.034** | **+0.142 ± 0.055** | **+0.267 ± 0.085** | 16.22 |
| claude-opus-4-8 | prior | **hard-20** | 20 | **0.751 ± 0.028** | +0.138 ± 0.058 | +0.286 ± 0.104 | 18.27 |
| claude-opus-4-8 | no-prior, sum-replace (free-form prompt) | hard-20 | 20 | 0.567 ± 0.037 | +0.088 ± 0.064 | +0.194 ± 0.079 | 13.08 |
| claude-opus-4-8 | no-prior, sum-augment (free-form prompt) | hard-20 | 20 | 0.630 ± 0.042 | +0.171 ± 0.048 | +0.260 ± 0.096 | 15.12 |
| claude-opus-4-8 | no-prior, sum-replace **(structured prompt)** | hard-20 | 20 | 0.618 ± 0.042 | +0.068 | +0.089 | 14.85 |
| claude-opus-4-8 | no-prior, sum-augment **(structured prompt)** | hard-20 | 20 | **0.725 ± 0.034** | +0.115 | **+0.318** | 17.77 |
| qwen3.5-4b | no-prior | **hard-20** | 20 | **0.270 ± 0.050** | +0.121 ± 0.056 | −0.039 ± 0.103 | 7.41 |
| qwen3.5-4b | prior | hard-20 | 20 | 0.295 ± 0.050 | −0.014 ± 0.048 | −0.065 ± 0.094 | 8.31 |
| qwen3.5-4b | no-prior, sum-replace (free-form prompt) | hard-20 | 20 | 0.181 ± 0.069 | +0.110 ± 0.044 | −0.112 ± 0.123 | 6.83 |
| qwen3.5-4b | no-prior, sum-augment (free-form prompt) | hard-20 | 20 | 0.222 ± 0.076 | −0.040 ± 0.105 | −0.236 ± 0.126 | 7.79 |
| qwen3.5-4b | no-prior, sum-replace **(structured prompt)** | hard-20 | 20 | **0.319 ± 0.062** | +0.098 | **+0.084** | 8.95 |
| qwen3.5-4b | no-prior, sum-augment **(structured prompt)** | hard-20 | 20 | **0.320 ± 0.060** | +0.031 | **+0.033** | 9.39 |
| qwen3-4b-Instruct-2507 | no-prior | **hard-20** | 20 | **0.260 ± 0.028** | +0.070 ± 0.040 | +0.113 | 6.43 |
| qwen3-4b-Instruct-2507 | prior | hard-20 | 20 | 0.224 ± 0.061 | +0.017 ± 0.040 | +0.189 | 7.05 |
| qwen3-4b-Instruct-2507 | no-prior, sum-replace **(structured prompt)** | hard-20 | 20 | 0.267 ± 0.026 | +0.041 ± 0.049 | +0.076 | 6.73 |
| _gpt-5_ | _(pending)_ | hard-20 | | | | | |
| _gemini-3_ | _(pending)_ | hard-20 | | | | | |

### Key findings (claude-opus-4-8)

1. **Meta-learns on hard episodes** — I_score is significantly positive in BOTH conditions
   (TMLR +0.27 / +0.29; robust +0.14, ~2.6 SE). The model improves across trials within an episode.
2. **Hard-set concentrates the signal** — vs random-20, hard-20 has lower performance (0.67 vs 0.71, less
   ceiling) and a STRONGER, clearly-significant I_score (robust +0.142 vs +0.057). Hard episodes de-saturate
   the meta-learning measurement.
3. **Prior info: a non-significant performance trend, no effect on learning** — on hard-20, prior info raises
   performance by **+0.077 (paired t=1.89, df=19, p≈0.07, 12/20 episodes higher → NOT significant at N=20)**
   and leaves I_score unchanged. This strong model meta-learns with or without the structural hints; larger
   N needed to firm up the prior-info effect.

### qwen3.5-4b (local, no-prior, hard-20)

A frontier/4B gap as expected: **performance 0.270** (≈40% of opus-4-8's 0.674 on the same episodes; vs
the ~0.5 memoryless-heuristic level in the original task this is below-heuristic). Cross-trial signal is
ambiguous — robust I_score +0.121 (~2.2 SE, marginal) but TMLR I_score ≈ 0 — so weak/unclear meta-learning.

**Serving requirement**: Qwen3.5-4B is a *thinking* model; its reasoning eats the token budget before the
`ACTION:` line → **73% invalid** by default. Fix = run with `--no-thinking` (disables thinking via
`chat_template_kwargs.enable_thinking=false`) → **0.2% invalid**. Served via `serve_qwen.sh` (sglang,
OpenAI-compatible, DP=4 across 4 GPUs, tp=1); eval connects with `--provider openai --base-url ... --no-thinking`.
(The parser also now tolerates `stone <0>` angle-bracket placeholder copies — model-agnostic, no effect on Claude.)

### qwen3-4b-Instruct-2507 — prompt iteration (training-target prompt, NOT a TMLR baseline)

We switched the RL model from Qwen3.5-4B (GDN, slow/OOM in Megatron) to **Qwen3-4B-Instruct-2507**
(standard attention, trains 7–15× faster, 0 OOM, ~3 min/rollout-step). Its zero-shot scores land right on
Qwen3.5's level (no-prior **0.260** ≈ qwen3.5 0.270, n.s.), so the eval is aligned. But output analysis
surfaced a behavioral blocker for the two-stream memory-RL:

- **It barely uses potions** — across a serious-setting training rollout (64 episodes × 10 trials, 20
  steps/trial): cauldron 1000 / end-trial 830 / **potion only 43 (2.3%)**; **60% of episodes use ZERO
  potions**. On hard episodes it just submits the *raw* starting stones and ends the trial.
- **Wrong mental model** (from eval transcripts): it believes a potion *adds* reward, so it either skips
  potions or dumps its highest stone into a random potion and destroys it (+15 → +1).
- **Consequence**: the WRITE stream is starved — no potion → no transitions → `n_fk≈0` at every rewrite
  point → **WRITE n = 0** (confirmed at steps5 AND steps20, so it is NOT a step-budget issue). ACT is fine
  (n=640, mean trial reward ≈ 7.7). This also caps ACT performance at ~26% of oracle.

**Fix being iterated**: append an exploration-guidance block to the system prompt that (1) states potion
effects are HIDDEN and reset per episode, (2) corrects the mental model — a potion TRANSFORMS a stone and
can raise OR lower reward, (3) tells it to experiment in early trials and exploit later, (4) tells it not to
just submit raw stones. This is a **leading, non-TMLR prompt** — it is the *training-target* prompt for the
RL run, kept separate from the TMLR baseline rows above. Offline eval (hard-20, no-prior, no-summary) is the
fast proxy; success = performance up AND potion-usage up from 2.3% (the latter directly predicts WRITE
revival). Injected via `eval_alchemy.py --extra-system-file <path>`; once chosen, baked into
`prompts_eval.py` so training (`alchemy_rollout.py`, which imports the eval prompt) inherits it — train/eval parity.

**Base prompt it starts from** (`build_eval_system(prior_info=False)` = `MAIN_PROMPT` + `FORMAT_BLOCK`,
verbatim TMLR Fig 5a + our format block) — the exploration block is APPENDED to this.

Candidate exploration blocks (`examples/alchemy/eval/prompt_variants/`):

**`explore_v1.txt`** (detailed):
```
IMPORTANT — how to actually play well:
- Each potion's effect is HIDDEN and is different every episode. You must EXPERIMENT to discover it.
- A potion does NOT simply add reward. It TRANSFORMS a stone's properties (colour/size/shape), which can RAISE or LOWER the reward. Some potions help, some hurt — you cannot know without trying.
- Strategy: in the EARLY trials, EXPERIMENT. Place stones into different potions and read the "Outcome stone" line to see how the properties and reward changed. Build up knowledge of what each potion does.
- The stones and the hidden chemistry are the SAME across all trials of an episode, so what you learn early lets you turn low-reward stones into high-reward ones in later trials.
- Do NOT just submit your starting stones to the cauldron. The highest scores come from transforming a stone with the right potion(s) BEFORE placing it in the cauldron.
```

**`explore_v2.txt`** (lean):
```
Note on strategy: each potion's effect is hidden and can either INCREASE or DECREASE a stone's reward — the only way to learn it is to try. Use the first few trials to experiment: place stones into different potions and observe the Outcome stone, then exploit what you learned in later trials. Transforming a low-reward stone into a high-reward one with the right potion scores far more than just submitting the stones you start with.
```

**[Update 2026-06-18] A/B result — the exploration prompt WORKS (wakes up potion use), but does not by
itself raise score.** Ran both variants × {baseline / summary-replace / prior} on hard-20 (no-thinking,
workers 8). The decisive metric is **potion-usage rate** (= potion actions / all actions; directly predicts
whether the WRITE stream gets transitions) alongside normalized performance:

| Condition | perf | I (robust) | I (TMLR) | **potion %** | zero-potion eps |
|---|---|---|---|---|---|
| **old baseline (NO explore block)** | 0.260 ± 0.028 | +0.070 | +0.113 | **6.2 %** | **13/20** |
| explore-v1, baseline | 0.244 ± 0.036 | +0.014 | −0.008 | **44.7 %** | 0/20 |
| explore-v1, sum-replace | 0.219 ± 0.041 | +0.056 | −0.051 | 38.4 % | 0/20 |
| explore-v1, prior | 0.260 ± 0.030 | +0.000 | −0.015 | 47.5 % | 0/20 |
| explore-v2, baseline | 0.237 ± 0.024 | +0.028 | +0.035 | 42.8 % | 0/20 |
| explore-v2, sum-replace | 0.240 ± 0.038 | +0.021 | +0.079 | 35.7 % | 0/20 |
| explore-v2, prior | 0.269 ± 0.036 | +0.052 | −0.011 | 46.9 % | 0/20 |
| _(reference) qwen3.5-4b baseline_ | 0.270 ± 0.050 | +0.121 | −0.039 | _53.1 %_ | _0/20_ |

Findings:
1. **Exploration revived**: potion-usage jumps **6.2 % → ~43–47 %** and zero-potion episodes go **13/20 → 0/20**
   — now on par with Qwen3.5 (53 %). v1 and v2 are equivalent (v1 explores slightly more, v2 scores slightly
   higher; all diffs within noise). This is exactly the blocker for the WRITE stream: explore ⇒ transitions ⇒ WRITE n > 0.
2. **Performance unchanged** (all 0.24–0.27, within SE; baseline even dips a hair). The model now *explores a
   lot but exploits poorly* — same pattern as `--prior-info` and as Qwen3.5 (which explores 53 % yet still
   scores only 0.270). Exploration ≠ knowing how to explore. Raising ACT score is the RL training job, not the
   prompt's; the prompt's job is only to make both streams produce data.
3. **Decision**: go with **explore-v2** (lean; less answer-leaking / closer to zero-shot; v2-prior 0.269 is the
   top cell). Next: bake the v2 block into `prompts_eval.py` so training (`alchemy_rollout.py`, which imports
   the eval prompt) inherits it (train/eval parity), then run a training rollout to confirm WRITE n rises off 0.

_Run dirs: `logs/eval-q34b-{v1,v2}-{baseline,summary,prior}/`. potion% = potion / (potion+cauldron+end-trial)._

### performance by difficulty (claude-opus-4-8, no-prior, random-20)

Normalization controls raw difficulty, yet the model is still relatively further from optimal on
more-bottlenecked (fewer-edge) episodes — the bottleneck reasoning is genuinely harder:

| graph edges | difficulty | N | performance |
|---|---|---|---|
| 7 | hardest (5 missing) | 9 | 0.653 ± 0.044 |
| 8 | (4 missing) | 2 | 0.398 |
| 9 | (3 missing) | 1 | 0.461 |
| 10 | (2 missing) | 6 | 0.864 ± 0.050 |
| 12 | no bottleneck | 2 | 0.930 |

(8/9-edge tiers are tiny-N noise; the well-sampled 7/10/12 tiers show the monotone trend.)

### Oracle achievable reward by difficulty (ALL 1000 episodes)

The exact oracle is now precomputed for **all 1000 episodes** (`oracle_cache.json`). Mean optimal raw
reward per trial (of 45 max), grouped by graph-edge count:

| graph edges | difficulty | N episodes | oracle reward/trial (of 45) | median | % trials solvable to 45 |
|---|---|---|---|---|---|
| 7 | hardest (5 missing) | 247 | 25.9 | 30 | 20% |
| 8 | (4 missing) | 174 | 27.8 | 31 | 24% |
| 9 | (3 missing) | 50 | 29.0 | 31 | 27% |
| 10 | (2 missing) | 278 | 29.9 | 31 | 31% |
| 12 | no bottleneck | 251 | 31.4 | 31 | 36% |
| **all** | | **1000** | **28.9** | — | 28% |

**Key intuition**: the bottleneck lowers the oracle's *achievable* reward only MILDLY — hardest→easiest
differ just ~5.5/45 (~21%); even the hardest 7-edge episodes are ~26/45 with 20% of trials fully solvable.
But the *model's* normalized gap by difficulty is much larger (0.65 on 7-edge vs 0.93 on 12-edge above).
⇒ **harder episodes are hard for the agent's inference/path-finding, NOT because the reward ceiling drops** —
exactly the property we want for isolating the meta-learning / memory challenge.

---

## Run artifacts

| Run | Dir (`examples/alchemy/logs/`) |
|---|---|
| opus-4-8 no-prior random-20 | `eval-claude-opus-4-8-noprior-20260613-161438/` |
| opus-4-8 no-prior hard-20 (merged) | `eval-claude-opus-4-8-noprior-hard20/` |
| opus-4-8 prior hard-20 | `eval-claude-opus-4-8-prior-20260613-190456/` |
| opus-4-8 sum-replace hard-20 (free-form) | `eval-claude-opus-4-8-sum-replace-noprior-20260614-145615/` |
| opus-4-8 sum-augment hard-20 (free-form) | `eval-claude-opus-4-8-sum-augment-noprior-20260614-151654/` |
| opus-4-8 sum-replace hard-20 (structured) | `eval-claude-opus-4-8-sum-replace-noprior-20260614-234739/` |
| opus-4-8 sum-augment hard-20 (structured) | `eval-claude-opus-4-8-sum-augment-noprior-20260614-234748/` |
| qwen3.5-4b sum-replace hard-20 (structured) | `eval-qwen3.5-4b-sum-replace-noprior-20260614-231736/` |
| qwen3.5-4b sum-augment hard-20 (structured) | `eval-qwen3.5-4b-sum-augment-noprior-20260614-232013/` |
| qwen3.5-4b no-prior hard-20 | `eval-qwen3.5-4b-noprior-20260613-223417/` |
| qwen3.5-4b prior hard-20 | `eval-qwen3.5-4b-prior-20260613-225217/` |
| qwen3.5-4b sum-replace hard-20 | `eval-qwen3.5-4b-sum-replace-20260614-010136/` |
| qwen3.5-4b sum-augment hard-20 | `eval-qwen3.5-4b-sum-augment-*/` |
| oracle cache (all 1000 episodes) | `examples/alchemy/eval/oracle_cache.json` |

**Prior-info effect (both models)**: NOT significant. opus-4-8 +0.077 (paired t=1.89, p≈0.07); qwen3.5-4b
+0.026 (paired t=0.34). Differs from TMLR's finding that prior info helped their (weaker) models. Caveat:
qwen's prior arm had 8.3% invalid (vs 0.2% no-prior) — the longer prior prompt degrades its format adherence.

**Summarization effect (BOTH models — contrary to TMLR)**: prompted cross-trial summary does NOT help
performance for either model; ranking is **no-summary > sum-augment > sum-replace** for both:
- Claude: 0.674 → augment 0.630 → replace 0.567. augment ≈ no-summary (Δ within ~1 SE, N=20); replace
  significantly lower (discarding the raw history loses info even for a model that summarizes well). One
  nuance: augment slightly *raises* Claude's I_score (robust +0.171 vs +0.142) — a better learning slope at
  a slightly lower level — but not performance.
- qwen3.5-4b: 0.270 → augment 0.222 → replace 0.181. Both hurt; augment also **collapses its I_score to
  −0.040** (and replace to a worse TMLR I_score) via REWRITE drift / catastrophic forgetting (`QWEN_SUMMARY_DRIFT.md`).

This is opposite to TMLR's "summarization unlocks meta-learning" — that likely held for weaker / shorter-context
models; modern long-context models already exploit the full no-summary history, so a self-written summary is at
best redundant (augment) and at worst lossy or drifting (replace). **Research framing for L_WM**: prompted
memory is neutral-to-harmful; the bar is a *trained* compressed memory (replace-style) that BEATS no-summary —
which prompted summary fails to do (replace is the worst cell for both models).

**[Update 2026-06-14] Structured (TMLR-format) summary prompt — the harm above was largely a PROMPT
artifact**: the runs above used a free-form prose summary prompt. Re-running qwen3.5-4b with a *structured*
prompt (TMLR's two sections — `### Potion Effects` + `### Highest Reward Combination` — plus an explicit
"carry forward confirmed effects; don't delete a rule just because a trial went badly" clause) erases the
harm entirely:
- **performance**: replace 0.181 → **0.319**, augment 0.222 → **0.320** — both now ON PAR with no-summary
  (0.270). Paired vs no-summary: replace +0.049 (t=0.97), augment +0.051 (t=1.00) — **not significantly
  different** (N=20). Paired new-vs-free-form: replace **+0.138, t=4.40, 17/20 higher (significant)**;
  augment +0.098, t=1.60 (directional).
- **drift fixed**: TMLR I_score flips positive (replace −0.112 → **+0.084**, augment −0.236 → **+0.033**);
  augment's robust-I_score collapse is repaired (−0.040 → +0.031). The ep55 smoking gun (`QWEN_SUMMARY_DRIFT.md`)
  no longer occurs — the +15 recipe (`blue small pointy + Orange → +15`) now survives M₁→M₉ instead of being
  overwritten by a "never transform" blob.
- **The earlier free-form analysis is not wrong** — it correctly characterized *that* prompt. The lesson is
  that prompted-summary quality is highly prompt-sensitive. **Revised L_WM framing**: a *well-prompted*
  replace-style memory (discard raw history, keep one compressed note) already MATCHES full history without
  loss for the 4B (0.319 vs 0.270, n.s.) — so the bar for a *trained* memory (L_WM) is to **significantly
  BEAT** no-summary, which even the good prompted summary does not (only reaches parity).

- **Claude opus-4-8, same structured prompt** — re-ran both modes (rows above). Same direction: the
  structured prompt lifts both, and **augment now lands ABOVE no-summary** (0.630 → **0.725** vs no-summary
  0.674; I_tmlr +0.260 → **+0.318**). Paired: **new-augment − free-form-augment = +0.095, t=2.29 (significant)**;
  new-augment − no-summary = +0.051, t=1.05 (n.s., i.e. augment ≥ no-summary). **replace** rises 0.567 → 0.618
  (t=1.32, n.s.) but stays **below** no-summary (−0.056 vs no-summary, n.s.) — for a strong long-context model,
  discarding raw history (replace) still costs more than it does for the 4B. Net: with a good (structured)
  prompt the "summary hurts" conclusion is overturned for **augment** in BOTH models (augment ≥ no-summary);
  **replace** reaches parity only for the 4B, not for Claude.

**Failure-mode analyses**: no-summary trajectories → `QWEN_FAILURE_MODES.md`; summary-condition REWRITE
drift (free-form prompt) → `QWEN_SUMMARY_DRIFT.md`.

_Last updated: 2026-06-14. All conditions (no-summary / summary-replace / summary-augment, no-prior/prior)
complete for opus-4-8 and qwen3.5-4b on hard-20; oracle precomputed for all 1000 episodes._


