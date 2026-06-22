# DRAFT — Why structured `replace` summaries hurt Claude-Opus on Symbolic Alchemy

**Status:** DRAFT (analysis only, read-only over logs; no eval re-run).
**Date:** 2026-06-15
**Source runs (examples/alchemy/logs/):**
- No-summary: `eval-claude-opus-4-8-noprior-hard20/`
- Structured `replace`: `eval-claude-opus-4-8-sum-replace-noprior-20260614-234739/`
- Same 20 hard episodes, paired by `episode_index`.

Headline means: **no-summary perf 0.674 vs replace perf 0.618** (replace is *worse* for this strong long-context model).

---

## Headline conclusion

For Claude-Opus, `replace` (write a per-trial summary, then **discard the raw history**) loses because the summary
**prematurely freezes wrong / over-confident negative conclusions** — most often tagging a potion as
**"no effect / inert / unidentified"** after testing it on a stone where that potion *happens not to apply*.

In Alchemy each potion only acts on stones with a specific feature (e.g. "turquoise: pointy→round" does nothing on a
stone that is already round). The first time the agent tries such a potion it often sees "no change," writes
**"inert"** into the summary, and — because the raw history is thrown away — **never re-tests it**. When a later trial
presents a stone in the state where that potion *would* fire (and would unlock a +15 chain), the `replace` agent has no
reason to try it and settles for a low-value bank.

No-summary keeps the full transcript, so it retains the *raw observation* ("turquoise had no effect *on a round stone*")
rather than the *lossy generalization* ("turquoise is inert"), and freely re-tests the potion on a pointy stone the next
time, discovering the +15 path.

This is the mirror image of the 4B result (4B: replace **0.319 ≥** no-summary **0.270**): a weak model can't exploit the
full history anyway, so a structured summary is a net scaffold; a strong model exploits the history very well, so
compressing it away discards evidence it would otherwise have used.

---

## Per-episode drop table (top replace losses)

| episode | no-summary perf | replace perf | drop (ns − rp) | where it breaks |
|--------:|----------------:|-------------:|---------------:|-----------------|
| 59 | 0.679 | 0.349 | **+0.330** | turquoise tagged "inert" → never re-tested on pointy stones |
| 26 | 0.742 | 0.443 | **+0.299** | orange (round→pointy) never recorded → blue-small-pointy +15 chain missed |
| 6  | 0.834 | 0.564 | **+0.270** | red/pink/green all marked "no observed effect" |
| 53 | 0.593 | 0.352 | **+0.241** | green/orange/red all marked "inert" |
| 30 | 0.805 | 0.598 | **+0.206** | pink/green/red all marked "no effect" |

Common signature: in every one of these episodes the two runs **track each other in the early trials and diverge in the
late trials (5–10)**, exactly when a previously-"inert" potion becomes the key to a high-value chain. (e.g. ep59 trial#5
ns 31 / rp 3; ep26 trial#8 ns 16 / rp 2; ep6 trial#9 ns 16 / rp 1; ep53 trial#10 ns 30 / rp 2; ep30 trial#10 ns 31 / rp 3.)

---

## Case 1 — EP 59, trial #5: "turquoise is inert" freezes the +15 enabler

This is the cleanest case. By trial 5 the `replace` summary had written, across multiple trials:

> **Turquoise**: No observed effect on a purple small round stone (reward stayed +1). … Appears inert.
> Note: No potion has yet been observed to change a stone's COLOR (blue→purple) or SHAPE (pointy→round).
> … only green and red have any effect, and both only change SIZE.

The true chemistry (revealed in the no-summary run) is **turquoise = pointy→round**, the move that lifts
`purple large pointy +1 → purple large round +15`. Turquoise had only ever been tried on *round* stones, where it
correctly did nothing — so the summary's "inert" was an over-generalization of a true-but-narrow observation.

**No-summary, trial #5** (oracle 31): after sizing up three purple stones to *large pointy +1*, the agent re-tests
turquoise on a **pointy** stone:

> ACTION: Place stone 2 in potion 0 (turquoise)
> OUTCOME: purple large round with reward +15
> "Turquoise changed stone 2 from pointy to round, raising reward from +1 to +15! Turquoise turns pointy→round."

It then turquoise-upgrades a second stone and banks **31**.

**Replace, trial #5** (same stones, three turquoise potions in inventory at slots 0/9/+): the agent reaches the identical
*three purple large pointy* state, but its summary says turquoise is inert and "no pointy→round potion exists," so:

> "There's still one green potion left, but green only sizes up small stones … No potion to change pointy→round.
> Bank all three stones at +1 each."
> ACTION: Place stone 0 / 1 / 2 in the cauldron → score **3**

Turquoise was sitting in the inventory, applicable to the exact stones present, and would have given +15 each — but the
frozen "inert" label removed any motive to try it. Net: **3 vs 31** on a trial the model fully solved when it had the
raw history. (Earlier trial #1 even shows the opposite sign: replace banked the pre-given +15 stones cleanly for 45,
ns 31 — so this is not a capability gap, it is an evidence-availability gap.)

## Case 2 — EP 26, trial #8: dropped "orange = round→pointy" breaks the blue-small-pointy +15 chain

Both runs hit the **identical** trial #8 (stones: purple small round −1, blue large round −1, purple large pointy −1;
potions include orange + green; top combo this episode = **blue small pointy +15**).

The `replace` summary (seen entering trial #8) records the chemistry partly right but has:

> **Orange**: No effect observed (tested on blue small round). Appears inert (or acts on a feature not present).
> **Turquoise**: No effect observed … Appears inert.

The actual mechanic (used explicitly in the no-summary reasoning) is **orange = round→pointy**.

**No-summary, trial #8** (score 16): the agent's opening reasoning lists the full map including
`orange(round→pointy)`, then runs `blue large round −1 →orange→ blue large pointy +1 →green→ **blue small pointy +15**`,
banks 15 (+1 from a second stone) = **16**.

**Replace, trial #8** (score 2): orange's true effect was never in the summary, so the agent never considers the
round→pointy step:

> "Best target is blue small pointy (+15), but I have no blue pointy stone here, and **no round→pointy potion**."
> → settles for `blue large round →green→ blue small round +1` and `purple large pointy →green→ purple small pointy +1`.
> Banks **2**.

The +15 path was reachable with potions in hand; only the *knowledge that orange turns round→pointy* was missing — and
it was missing **only because the summary had discarded the raw observation and labeled orange "inert."**

## Cases 3–5 — same mechanism (corroborating)

- **EP 6** — final summary: "**Red**: No observed effect…", "**Pink**: No observed effect…", "**Green**: No observed
  effect on any stone tested." Three potions written off. ns vs rp diverges hard at trials 5/7/9 (ns 17/16/16 vs
  rp 3/2/1) — late trials needing one of the written-off potions.
- **EP 53** — final summary: "**Green**: Appears **inert**", "**Orange**: Appears **inert**", "**Red**: Appears inert
  on purple stones." Trials 3/5/6/7/10 all collapse (e.g. trial#10 ns 30 vs rp 2).
- **EP 30** — final summary: "**Pink**: No effect observed", "**Green**: No effect observed", "**Red**: No effect
  observed." Trials 5/7/9/10 collapse (trial#10 ns 31 vs rp 3).

In each, the dropped potions are precisely the ones whose effect is *state-conditional* (only fires on a feature not
present when first tested), so a single "no change" observation got over-generalized to "inert," frozen into the
summary, and never revisited once the raw history was gone.

---

## Mechanism summary

1. **The loss is information loss, not reasoning loss.** Replace matches no-summary on early trials and on trials the
   summary happens to cover; it fails specifically when a trial needs a fact the summary *over-compressed away*. The
   replace agent's per-turn reasoning is sound given its (impoverished) summary.

2. **The lossy step is over-generalizing a narrow observation into a global negative.** "Potion X did nothing *on this
   particular stone*" (true) → "Potion X is inert" (false). The raw transcript preserves the qualifier; the summary
   drops it. Because state-conditional potions are the norm in Alchemy, these false "inert" verdicts are common and
   high-cost (each blocks a +15 chain).

3. **Discarding history removes the corrective.** With full history Claude re-tests dubious potions (it literally says
   "let me re-test turquoise on a pointy stone"); with replace it has no record that the earlier test was on the wrong
   state, and the confident summary suppresses re-testing.

4. **Contrast with the 4B model (replace 0.319 ≥ no-summary 0.270).** A weak model can't reliably mine a long raw
   transcript, so a structured summary is net-positive scaffolding even when lossy. A strong long-context model already
   mines the raw transcript near-optimally, so replacing it with a lossy summary is strictly subtractive — you trade
   high-fidelity evidence the model *was* using for a confident-but-wrong abstraction. Hence the crossover: the same
   `replace` mechanism that lifts 4B *lowers* Claude-Opus.

**Implication for design:** for strong models, prefer `augment` (summary on top of raw history) over `replace`, or make
the summary explicitly *uncertainty-preserving* — record "tested only on round stones, effect on pointy unknown" instead
of "inert," so later trials still have a reason to probe.
