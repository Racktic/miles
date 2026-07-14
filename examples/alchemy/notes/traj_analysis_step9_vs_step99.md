# Symbolic Alchemy — Eval Trajectory Analysis: step 9 vs step 99

**Run:** `qwen3-4b-curr950-actonly-w3-expl03-budgetv3-r120-e10-20260630`
(Qwen3-4B-Instruct-2507, ACT-only, explore reward β=0.3, budget reminder + explore_v3 prompt, memory window 3)
**Compared:** eval on `hard20` (20 episodes × 10 trials), early checkpoint **rollout_9 (step 9)** vs late **rollout_99 (step 99)**
**Dirs:** `.../traj/eval/hard20/rollout_9` vs `.../rollout_99`
**Method:** 3 parallel read-only analysis agents (quantitative stats / written-memory / behavior-patterns) + a same-episode verbatim memory comparison.

---

## TL;DR (integrated synthesis)

Over training (step 9 → 99), the model became a **much more exploratory policy that actually discovers the environment's potion→stone transformation rules**, and mean per-trial score rose **+54% (8.29 → 12.77)**. All three analysis angles converge on the same story, and the improvement shows up in every trial index (not just late trials).

### Headline shifts (cross-validated across the 3 analyses)

| dimension | step 9 (early) | step 99 (late) | change |
|---|---|---|---|
| mean per-trial score | 8.29 | 12.77 | **+54%**, all 10 trials up |
| distinct potions tried / trial | 2.58 | 5.81 | **+125%** (exploration doubled) |
| turns / episode | 53.8 | 94.1 | **+75%** |
| potions credited with a REAL effect in final memory (per ep) | 0.20 | 1.00 | **~5×** |
| memories with an actionable replication plan | 0.01 | 0.12 | 12× |
| invalid-action rate | 1.7% | 6.9% | ~4× (the cost) |
| distinct stones touched / trial | 2.55 | 2.86 | ≈ unchanged |

### What changed (narrative)
1. **Exploration surged, and almost entirely in *potion* space** — distinct potions/trial more than doubled while stone coverage barely moved; the agent learned to try many more transformations on the *same* stones. The β=0.3 explore reward is clearly biting.
2. **Memory went from a "catalog of null results" to a "causal rule-set."** Length, formatting, and hedging were already saturated at step 9 and barely changed — the improvement is entirely in *substance*: ~5× more potions credited with a real, directional effect, and 12× more actionable replication plans.
3. **The defining *behavioral* difference is late-trial behavior.** Step 9 exploits-and-quits (late trials: ~1.7 distinct potions, ~4.6 turns — bank the obvious +15 and end). Step 99 exploits-then-keeps-probing (~4.5 distinct potions, ~8.3 turns — bank the +15, then spend remaining turns searching for a higher transform). Reasoning shifts from generic color/shape heuristics to compositional transformation chains with potion-depletion tracking.

### Caveats worth flagging (for the research story)
- **The gain is mostly a higher *baseline per trial*, not steeper *in-episode* learning.** The crude within-episode improvement proxy `mean(trials 5-9) − mean(trials 0-4)` barely moved (+2.31 → +2.87). Training made the model uniformly better + more exploratory; the "memory visibly compounds within an episode" slope did **not** get much steeper. This is relevant to the memory-RL thesis.
- **The ~4× invalid-action rate is the quality cost of harder exploration** (illegal "put" attempts on spent/incompatible potions, plus a few empty outputs) — but it is **not** degeneracy: no repeat-loops (consecutive-repeat ≈ 0), and trials still self-terminate before the 20-turn cap.

---

## Concrete memory examples (same episode, step 9 vs step 99)

These make the "content quality" shift tangible. Same episode, same hidden rule — only the checkpoint differs. Shown is the **final written memory** (the culmination of what the agent learned that episode).

### Episode 30 — the clearest case (step 9 total **−6** → step 99 total **100**)

- step 9 per-trial: `[-2, -1, 0, 3, 1, 1, -1, -6, 1, -2]` (never found anything; net negative)
- step 99 per-trial: `[1, 2, 1, 17, 15, 17, 16, -1, 16, 16]` (found the +15/17 transforms)

**step 9 final memory** — concludes there is *no* useful transformation and even dismisses real effects as "system errors":
> - **Green**: No effect on reward or stone properties; may cause isolated, inconsistent changes … **but these appear to be system errors or artifacts. Verified as inert** in consistent testing …
> ### Highest Reward Combination
> +15 reward achieved with blue small round stones **in untransformed form** … The highest reward is thus **achieved only by preserving the blue small round stone in its original form.**

**step 99 final memory** — discovered **two** real transformation paths to the +15 stone:
> - **Turquoise**: Transforms purple small round into **blue small round with reward +15**; transforms purple large round into blue large round with reward +1 …
> - **Red**: Converts blue large round into **blue small round with reward +15**; a previously unobserved transformation that produces the highest known reward.
> ### Highest Reward Combination
> +15 … achieved by blue small round stones (**formed via transformation of either purple small round using turquoise or blue large round using red**) …

→ Step 9 concluded "don't touch the good stone" (a failure to explore, stated with confidence); step 99 learned *how to manufacture* the good stone from two directions.

### Episode 64 — "max is +1" vs the real +15 rule (step 9 total **30** → step 99 total **159**)

- step 9 per-trial: `[2, 2, 3, 2, 0, 17, 2, -2, 3, 1]`
- step 99 per-trial: `[30, 17, 30, 31, 2, 17, 2, 0, 30, 0]`

**step 9 final memory** — wrongly caps the reward at +1 and dismisses the +15 it saw:
> ### Highest Reward Combination
> +1 reward achieved … This is the **maximum reward achievable** across all tested transformations … **No transformation increases reward beyond +1.** A single purple small round stone with reward +15 is observed as an exceptional high-value stone, but **this does not exceed the inherent maximum reward structure** … the highest *achievable* reward after transformation **remains +1.**

**step 99 final memory** — found the +1→+15 transformation rules:
> - **Pink**: Converts purple small pointy stones into purple small round stones with a **reward increase from +1 to +15**. Effective only on purple small pointy stones …
> - **Orange**: Transforms blue large round into blue small round (−1 → +1); converts **purple large round into purple small round with reward +15** …
> ### Highest Reward Combination
> Purple small round with reward +15, **achieved through transformation via orange (from purple large round) or pink (from purple small pointy)** …

→ Same environment: step 9 declares "+1 is the ceiling" (under-exploration → wrong world-model); step 99 found the concrete recipes that reach +15.

**Pattern:** the early model's confident-but-wrong "nothing works / this is the max" is *under-exploration dressed as certainty*; the late model's memory is grounded in observed, directional, replicable transitions — and the reward follows.

---

## Memory quality — one worked example per characteristic

Each of Agent 2's memory-quality findings, shown on the single episode that most cleanly isolates it (final written memory; **bold** = the tell).

### ① Catalog of null-results → causal rule-set — **episode 15** (step 9 total 99 → step 99 total 145)

The marker scan makes this the cleanest case: step 9 encodes **0** causal rules, step 99 encodes 6.

**step 9** — all six potions recorded as "no effect", concludes nothing transforms:
> - Pink: **No effect on any stone tested; does not change reward** (tested on blue small round, blue large round, purple large pointy …)
> - Green: **No effect** … / Yellow: **No effect** … / Red: **No effect** … / Orange: **No effect** … / Turquoise: **No effect** …
> ### Highest Reward Combination
> … **All tested transformations resulted in no change to reward value.**

**step 99** — every potion carries a directional, reward-annotated rule:
> - **Red: Transforms small blue stones into large blue round stones; increases reward from -1 to +1** … transforms purple small round stones into purple large round stones **with reward preserved at +15** …
> - **Green: Converts blue large round stones into blue small round stones with a reward decrease from +1 to -1** … reduces purple large round to purple small round **with reward dropping from +15 to +1** …

### ② Passive description → actionable replication plan — **episode 1** (step 9 total 28 → step 99 total 113)

**step 9** — notes the good stone exists, but only passively ("just don't touch it"):
> The highest reward … is +15, achieved by blue small pointy stone. **This stone maintains its reward at +15 regardless of potion interaction** …

**step 99** — gives an explicit multi-step recipe to *manufacture* it:
> Blue small pointy stone with reward +15 … **This can be replicated by first converting purple large pointy stones into blue large pointy stones via orange, then transforming them into blue small pointy stones via pink.**

### ③ Spurious confident assertion (under-exploration dressed as certainty) — **episode 64** (step 9 total 30)

step 9's memory here has 4 sweeping assertions, **all false** — step 99 on the same episode found +15/+30 transformation paths and scored 159:
> +1 … is the **maximum reward achievable** across all tested transformations and stone types … **No transformation increases reward beyond +1.** A single purple small round stone with reward +15 is observed …, but **this does not exceed the inherent maximum reward structure**. The +15 value … **is not a result of any transformation**; the highest *achievable* reward after transformation **remains +1.**

**Takeaway:** the early failure mode isn't uncertainty — it's *confident wrong closure* from under-exploration ("nothing works / +1 is the ceiling"). The explore reward pushes the late model to keep probing until the memory reflects the true, replicable transformation rules — and reward follows.

---

## Agent 1 — Quantitative stats (full)

The env uses two action types only: potion-application ("put stoneX into potionY") and a voluntary "end the trial" (one per trial). No separate submit/cauldron action.

| Metric | EARLY (step 9) | LATE (step 99) | Δ (LATE−EARLY) |
|---|---|---|---|
| **Overall mean per-trial score** | **8.29** | **12.77** | **+4.48 (+54%)** |
| Trial-score trajectory (idx 0→9), EARLY | [8.4, 7.4, 6.9, 7.0, 6.0, 7.7, 10.1, 13.7, 7.4, 8.5] | | |
| Trial-score trajectory (idx 0→9), LATE | | [10.2, 9.0, 12.3, 11.9, 13.4, 12.5, 13.7, 16.7, 13.3, 15.0] | per-trial +1.8…+7.4 |
| Improve proxy mean(5-9)−mean(0-4) | +2.31 | +2.87 | +0.56 |
| **Mean turns / episode** | **53.75** | **94.05** | **+40.3 (+75%)** |
| Mean turns / trial | 5.38 | 9.41 | +4.03 |
| Action mix: potion-apply | 875 (81.4%) | 1668 (88.7%) | +7.3 pp |
| Action mix: end-trial (voluntary) | 200 (18.6%) | 198 (10.5%) | −8.1 pp |
| **Invalid-action rate (valid==False)** | **1.67% (18)** | **6.91% (130)** | **+5.24 pp (~4×)** |
| **Distinct potions tried / trial** | **2.58** | **5.81** | **+3.23 (+125%)** |
| Distinct stones touched / trial | 2.55 | 2.86 | +0.31 |

Per-trial turns (idx 0→9): EARLY [10.2,6.1,5.0,4.3,4.5,5.2,4.8,4.2,4.8,4.8] → LATE [13.9,10.4,8.9,8.8,8.6,9.4,9.3,7.7,9.0,8.3]; more turns on every trial index, largest absolute gains mid-episode.

Takeaways: (1) score +54% spread across all 10 trials (competence lift, not only better in-episode adaptation — the improve proxy barely moved). (2) far more exploratory (potions/trial +125%, turns/ep +75%). (3) invalid rate ~4× — the quality cost of heavier exploration. (4) voluntary end-trial got rarer (stops bailing early). (5) added exploration is almost entirely in potion space, not stone space.

---

## Agent 2 — Written-memory evolution (full)

Potions are referred to by color name (not index). 20 episodes × up to 9 written summaries (window-3 working memory).

| Metric | EARLY (step 9) | LATE (step 99) |
|---|---|---|
| Mean chars / words per summary | 1565 / 251 | 1614 / 259 |
| Length trial0 → trial8 (chars) | 981 → 1799 (grows) | 1092 → 1904 (grows) |
| Frac using bullet/section formatting | 1.00 | 1.00 |
| Frac hedging (unknown/unclear) | 0.01 | 0.01 |
| Frac asserting (always/confirmed/best) | 0.34 | 0.14 |
| Mean "no-effect / unchanged" claims per summary | 5.51 | 4.92 |
| Mean reward-delta mentions (from X to Y / inc/dec) | 3.74 | 5.17 |
| Frac with actionable replication plan | 0.01 | 0.12 |
| **Potions credited with a REAL effect in final memory (per ep)** | **0.20** | **1.00 (~5×)** |
| Episode total reward (mean) | 82.8 | 127.7 (+54%) |
| Final-trial score (mean) | 8.4 | 14.9 (+77%) |

Length, formatting, hedging are essentially unchanged — the shift is entirely content quality:
- **EARLY = catalog of null results; LATE = causal rule-set.** Same `### Potion Effects` / `### Highest Reward Combination` scaffolding (learned before step 9), but EARLY mostly writes "no effect / unchanged" and never recovers the transformation graph; LATE encodes directional potion→stone rules with reward deltas (~5× more potions credited with a real effect).
- **LATE became actionable/plan-oriented** (0.01 → 0.12): chains steps ("first convert purple large pointy → blue large pointy, then apply pink"), vs EARLY describing stones as passively "stable / unaffected."
- **Spurious generality dropped (assert 0.34 → 0.14):** EARLY over-asserts sweeping claims ("no combination improved the reward") that are really failures to explore; LATE asserts less globally but grounds each claim in an observed transition.
- Memory quality co-moves with reward (episode total 82.8 → 127.7; final-trial 8.4 → 14.9).

*Note:* absolute "potions with real effect" counts are conservative (strict regex); the ~5× relative gap is the robust signal, corroborated by reward-delta and score metrics.

---

## Agent 3 — Behavior / action patterns (full)

Both checkpoints show the same qualitative explore→exploit gradient (more potion-testing in trials 0-2, more submitting in trials 7-9), but step 99 has a substantially richer world-model and a deeper, *sustained* exploration policy. Neither degenerates into repeat-loops; neither burns the 20-turn cap.

| Metric | step 9 early (tr0-2) | step 9 late (tr7-9) | step 99 early | step 99 late |
|---|---|---|---|---|
| turns / trial | 7.08 | 4.60 | 11.05 | 8.32 |
| potion applies / trial | 4.27 | 1.88 | 7.83 | 4.78 |
| distinct potions tried / trial | 4.23 | 1.67 | 7.58 | 4.47 |
| submits / trial | 1.82 | 1.72 | 2.22 | 2.32 |

Whole-episode: mean num_turns 53.75 → 94.05; 20-turn cap hits 0/200 (step9) vs 2/200 (step99); consecutive-repeat loops ≈ 0 on both; invalid 1.7% → 6.9%.

- **Biggest behavioral difference = late-trial policy.** step 9 collapses to near-pure "submit the obvious +15 and end" (~1.7 distinct potions late); step 99 banks the +15 but keeps probing for a higher transform (~4.5 distinct potions late) — reward-positive here (tr7 16.65 vs 13.65; tr9 14.95 vs 8.45).
- **Both explore→exploit within an episode, different slopes:** distinct-potions/trial falls 4.23→1.67 (step 9, 2.5×) vs 7.58→4.47 (step 99, 1.7× — tapers, doesn't shut off).
- **Reasoning: generic → causal/compositional.** step 9 uses vague heuristics ("turquoise may not be effective for round stones"); step 99 reasons about transformation chains ("first purple large pointy → blue large pointy using orange, then → blue small pointy using pink"), tracks potion depletion ("pink is now unavailable, used up"), and preserves a high-value stone while testing others.
- **Extra turns are targeted, not wasted:** no consecutive-repeat loops (mean max consec 1.0 vs 1.1); higher invalid rate comes from aggressive attempts on depleted/incompatible potions, not stuck loops.
- **Submits/trial barely change** (~1.8→2.2 early, ~1.7→2.3 late); the reward gain is about *which stone / how transformed*, not submitting more often.

Verbatim reasoning snippets:
- EARLY (step 9), late trial — shallow one-shot exploit, ends in 2 turns:
  > "The blue small pointy stone yields a fixed +15 reward and is immune to all potion effects. Since no potion increases reward beyond this value … investing further turns … would not provide higher returns … ACTION: End the trial"
- LATE (step 99), same situation — banks the +15 but keeps probing with a compositional model:
  > "first converting purple large pointy stones to blue large pointy stones using orange, then transforming them into blue small pointy stones using pink …" … "With 17 turns remaining, I can continue exploring to see if any other transformations can yield a stone with even higher reward."

---

## Methods & paths
- Data: `.../logs/qwen3-4b-curr950-actonly-w3-expl03-budgetv3-r120-e10-20260630/traj/eval/hard20/{rollout_9,rollout_99}` — 20 episode dumps each; same `episode_index` set on both sides.
- Per-dump keys used: `per_trial_scores` (10), `num_turns`, `summaries` (written memory M_k), `turns[]` (`trial`,`action`,`action_int`,`valid`,`reward`,`raw_act`,`summary_in`).
- 3 read-only agents (quant / memory / behavior); verbatim examples pulled from the final non-empty `summaries` entry per episode. No files or wandb modified.
