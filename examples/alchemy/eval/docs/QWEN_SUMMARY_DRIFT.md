# Qwen3.5-4B on Symbolic Alchemy — Summary-Condition Failure (REWRITE drift)

Companion to `QWEN_FAILURE_MODES.md`. That doc covers the **no-summary** runs; **this doc covers the
SUMMARY experiment** — where the model writes a summary at each trial boundary (the REWRITE step) and acts
on it in later trials (TMLR A.4.2). Runs: `eval-qwen3.5-4b-sum-replace-*`, `eval-qwen3.5-4b-sum-augment-*`
(hard-20, no-prior). **Do not attribute these findings to the no-summary run.**

## Result: prompted summarization HURTS qwen3.5-4b

| condition | normalized performance | I_score (robust) | invalid% |
|---|---|---|---|
| no-summary (baseline) | 0.270 ± 0.050 | +0.121 ± 0.056 | 0.2% |
| summary-replace (summary REPLACES history) | 0.181 ± 0.069 | +0.110 ± 0.044 | 1.5% |
| summary-augment (summary ON TOP of history) | 0.222 ± 0.076 | −0.040 ± 0.105 | 0.3% |

Both summary modes are **≤ no-summary**; replace is significantly worse. This is the **opposite** of TMLR's
headline (summarization *unlocks* meta-learning) — but TMLR's result is for capable models (Gemini 2.5 etc.)
that write good summaries. The 4B does not.

## Headline: it's NOT a summarization-skill failure — the REWRITE chain DRIFTS

The summaries are detailed and structured (across the 180 replace-run summaries: **79% contain concrete
colour→effect rules**, mean length **1439 chars** — not lazy/empty). The model can even write the *correct*
recipe early. The failure is that the **iterated REWRITE over-weights the most recent (failed) trial and
progressively overwrites the correct rule, collapsing into a self-reinforcing pessimistic "never transform"
policy** — catastrophic forgetting through the summary chain. ~**88% of summaries** end up containing
"don't transform / potions don't help"-type language.

### Smoking gun — ep55 summary chain (M₁ → M₅ → M₉)

The model scored 15 in trial 1, so it *did* craft a +15. Watch its written memory degrade:

- **M₁ (after trial 1) — CORRECT:** "Transformation Rules Learned: **Purple small round + Pink Potion →
  blue small round (+15)**. *(This is the highest yielding transformation found)*." ← it found the winning recipe.
- **M₅ — starting to collapse:** "`blue small round`: Primary Win Condition. **Never transform**. … `blue
  large pointy` (-1): **NO TRANSFORMATIONS. Do not use any potion**." ← the recipe is gone; concludes potions don't work.
- **M₉ — fully collapsed:** "`blue small round`: Strategy: Identify immediately and pick. **Do not use
  potions.** Junk/Disposal Stones — Purple Stones: **NEVER TRANSFORM**, even negative ones pick as-is." ←
  a blanket "don't transform anything" policy; the M₁ recipe (Pink → +15) is permanently washed out.

## Mechanism

1. The REWRITE conditions on (previous summary + this trial's events). When a trial goes badly — often
   because the model *didn't* apply the recipe and saw mostly no-ops — the REWRITE over-weights those recent
   failures and rewrites the rule toward "potions don't help."
2. **Negative feedback loop:** pessimistic summary → model stops transforming → sees no transformations →
   summary becomes more pessimistic → … . Once it tips to "never transform," it can't recover.
3. **`replace` mode amplifies it (and is why replace < augment < no-summary):** the summary is the SOLE
   carrier of cross-trial info, so once it drifts the raw evidence (M₁'s recipe) is gone and the drift is
   irreversible. In `augment` the full history remains, so the bad summary mostly just adds noise.

## Relation to the no-summary analysis

Same underlying tendency — **over-weight recent failures → over-generalize to pessimism → drop the correct
rule** — expressed through different mechanisms:
- **no-summary:** in-the-moment over-generalization ("tried a few potions, they won't help" → give up;
  `QWEN_FAILURE_MODES.md` mode 5). No written memory.
- **summary:** the same pessimism gets **written into memory and persisted/compounded** across trials (this doc).

## Why this motivates L_WM

The prompted REWRITE drifts precisely because **nothing constrains the summary to stay accurate**. L_WM adds
exactly that signal: the memory must **predict the next transition d_t**. A "never transform / potions don't
help" memory cannot predict the +15 outcome the model actually observed → it is penalized and won't survive;
the `w_t` gate (penalize "seen-but-not-remembered") directly punishes forgetting M₁'s recipe. So the drift
observed here is direct evidence for training the memory (L_WM) rather than relying on prompted summaries.

_Source: `eval-qwen3.5-4b-sum-replace-20260614-010136/`, `eval-qwen3.5-4b-sum-augment-*`. Last updated 2026-06-14._

---

## [Update 2026-06-14] A structured summary prompt removes the drift (the analysis above stands for the free-form prompt)

Everything above describes runs with a **free-form prose** summary prompt. The drift it documents is real and
correctly attributed *to that prompt*. We then re-ran with a **structured** prompt mirroring the TMLR summary
example — two fixed sections (`### Potion Effects`, `### Highest Reward Combination`) plus an explicit
anti-forgetting clause ("carry forward everything already confirmed; only revise an entry when THIS trial gives
direct evidence against it; do NOT delete a confirmed potion effect just because a trial went badly"). **The
drift does not occur with the structured prompt.**

| condition (qwen3.5-4b, hard-20, no-prior) | performance | I_score (robust) | I_score (TMLR) |
|---|---|---|---|
| no-summary (baseline) | 0.270 | +0.121 | −0.039 |
| sum-replace, free-form (above) | 0.181 | +0.110 | −0.112 |
| sum-augment, free-form (above) | 0.222 | −0.040 | −0.236 |
| **sum-replace, structured** | **0.319** | +0.098 | **+0.084** |
| **sum-augment, structured** | **0.320** | +0.031 | **+0.033** |

Both structured modes climb back to parity with no-summary (paired vs no-summary n.s.; paired
structured−free-form for replace is +0.138, t=4.40, significant), and the negative TMLR I_score flips positive
— the signature of the drift (a learning curve that goes *down* across trials) is gone.

### Same ep55, now under the structured prompt — the +15 recipe SURVIVES

Re-reading the ep55 summary chain (`eval-qwen3.5-4b-sum-replace-noprior-20260614-231736/traj/ep55.json`):

- **M₁:** finds Green (small round → large round, +10) and Pink (large pointy → small round, +5) effects.
- **M₅:** discovers the winning recipe and **writes it down**: "`blue small pointy` + Orange → `blue small
  round` with a reward of **+15**".
- **M₉ — recipe RETAINED:** still lists "Orange: acts on blue stones to turn pointy→round; `blue small pointy`
  + Orange → `blue small round` (+15)". No "never transform" blanket; the +15 rule is carried to the end.

Contrast the free-form M₉ above ("NEVER TRANSFORM, even negative ones pick as-is"), where the same recipe was
washed out. The structured sections + carry-forward clause are what stop the recent-failure over-weighting from
overwriting confirmed rules.

### What this means for L_WM (refined, not reversed)

The drift was **prompt-fixable**, so "prompted memory inevitably drifts" is too strong. But the deeper point
for L_WM is intact and sharper: even the *good* prompted memory only reaches **parity** with keeping the full
history (0.319 vs 0.270, n.s.) — it does not beat it, and getting there required hand-engineering the prompt
per model. L_WM's target is a **trained** memory that (a) needs no prompt hand-tuning and (b) *significantly
beats* no-summary by being optimized to predict the next transition d_t — a signal a prompt cannot supply.
