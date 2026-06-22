# Qwen3.5-4B on Symbolic Alchemy — Failure Mode Analysis

Analysis of all 20 hard-20 trajectories (`eval-qwen3.5-4b-noprior-20260613-223417/`), no-prior, by 3
independent passes. Normalized performance 0.270 (vs opus-4-8 0.674 on the same episodes).

## Headline

**State tracking is fine; the bug is in the DECISION layer + rule-induction.** Qwen reads the obs
correctly, doesn't reuse consumed potions/stones (invalid ≈ 0), and ends trials sensibly. It also *tries*
to meta-learn (references "previous trials", forms hypotheses keyed on potion **colour** — the correct
transferable representation). But its decisions and induced rules are systematically wrong, so almost none
of the learning converts to score.

## Top failure modes (ranked by impact, convergent across all 3 passes)

All cases below are **verified** (programmatic detection + manual reading of the cited turn in
`eval-qwen3.5-4b-noprior-20260613-223417/traj/`).

1. **Stops at +1, never chains to +15** (most common; the dominant cause of low scores). Finds a single
   potion's effect, gets a stone to +1, declares "no further improvement" and banks +1 — but the oracle
   needs 2-3 chained transforms. It learns *isolated single-step rules*, not the compositional chemistry.
   _ep1 tr0 t12:_ REASONING "attempted all available potion combinations on the negative stones without
   success. The only positive reward stone is Stone 1 (+1). The negative stones … cannot be [improved]"
   → ACTION banks the +1. (ep53: 8/10 trials never exceed +1.)

2. **Over-explores and destroys a crafted +15 (fails to bank)**. Crafts a +15 stone, *recognizes* it, then
   keeps experimenting and degrades it instead of banking.
   _ep16 tr3 t6→t7:_ t6 outcome = `purple small round with reward +15` and it writes "I now have the
   high-value stone …"; t7 action is `Place stone 0 in potion 7` (does NOT bank → loses the +15).

3. **Banks negative stones instead of ending the trial**. When stuck at negative, dumps -1/-3 into the
   cauldron (scoring negative) rather than End (which gives 0). Treats "clear the stones" as an obligation.
   _ep11 tr1 t22:_ REASONING "no positive stones and no potions left …" → ACTION `Place stone 2 in the
   cauldron` (scores -1; End would have been 0). (ep37: repeated negative-banking, two trials score -3.)

4. **[RETRACTED] Outcome perception is actually CORRECT.** An earlier pass claimed ~16% of outcomes were
   misread as "no change". On verification this does NOT hold (programmatic scan = 0 hits; manual check of
   the cited ep64/ep55 turns): when Qwen says "no change" the outcome echo confirms it was a genuine **no-op**
   (potion blocked by the bottleneck, or direction already satisfied), and when a stone does change Qwen
   reads it accurately (_ep55 t4:_ "color changed from purple to blue … reward improved from -3 to -1").
   So single-step outcome reading is fine — consistent with the headline. The failures are in DECISION and
   rule INDUCTION, not perception.

5. **Over-generalizes a wrong rule → abandons experimentation**. From a few no-ops (real bottleneck),
   concludes "potions won't help" and stops trying — meta-learning *backfires*.
   _ep11 tr5 t49:_ REASONING "tried multiple green potions on Stone 2 without success. **It is likely any
   remaining potion will fail or possibly worsen it further**" → gives up and banks. (ep53 tr3-9: same
   pattern, perf 0.19, the worst episode.)

6. **Hallucinated game priors**. Invents mechanics that don't exist ("combine/mix stones", "multiplier",
   "completing sets", "in this game logic colour X reacts well to…"), wastes steps. _ep15, ep41, ep30._

7. **Gives up the key early exploration trials**. Ends trials 1-2 immediately ("rules never provided, I
   can't increase score → end"), closing the meta-learning window before it opens. _ep16 tr1/tr2._

8. **"Hallucinated full episode" (format break, RL-critical)**. _ep6 (perf 0.006)_: writes an entire
   fabricated multi-step rollout in ONE message and self-reports "Total score: 45". The harness only
   executes the first action; the model believes it won. **Lesson for RL/training: never trust the
   assistant's self-reported OBSERVATION/score — only the environment's returned reward/state.**

9. **Step budget eaten by exploration → no steps left to bank**. Fiddles with negative stones, then runs
   out of the 20 steps before banking the +15. _ep15 tr7 textbook._ A trivial "bank all non-negative
   stones first, then experiment with leftover steps" heuristic would lift scores a lot.

## Meta-learning verdict

**Weak, unstable, often counterproductive** — consistent with the ambiguous I_score (robust +0.121
marginal, TMLR ≈ 0). Evidence it tries (ep9, ep55, ep59 reuse colour-keyed rules across trials and improve);
evidence it fails (over-generalizes ignoring feature-dependence + bottleneck, learns wrong/hallucinated
rules, or learns "nothing works" and quits). The transferable *representation* (potion colour) is right;
the *induction* and *exploitation* are broken.

## Actionable takeaways

- The biggest wins are **decision-layer heuristics**, not better state tracking: (a) bank a +15 the instant
  you have it, (b) never cauldron a negative stone — End instead, (c) chain transforms toward the target,
  don't stop at +1.
- For RL training on this env: the env reward is the only ground truth; ep6 shows the model will
  confabulate an entire successful episode in-context.
