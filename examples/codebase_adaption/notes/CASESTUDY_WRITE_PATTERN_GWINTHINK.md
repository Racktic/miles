# WRITE behavior case study — `smith-4b-v3nocurr-explore-gwin-think`

Data: `/project/flame/qixinx/backups/smith-4b-v3nocurr-explore-gwin-think-resume/traj/train/rollout_{108..190}`,
83 rollouts x 16 episodes x 12 WRITEs = **15,936 WRITE records**. Windows: early 108–125 (3,456),
mid 140–160 (4,032), late 175–190 (3,072). All numbers recomputed from `write_audit` /
`summaries` / `outcomes` / `act_explore` with stdlib only; the format gate was re-implemented from
`codebase_advantage.memory_format_ok` and reproduces the logged `format_ok` on **10,226/10,226**
trained writes exactly.

Two conventions matter. (a) `format_ok` is only stamped on *trained* WRITE samples (61–64% of all
writes; the episode-tail write has no downstream window, and ~25% more are dropped by
`truncate_samples_by_total_tokens`), so all gate/reward rates below are over trained writes.
(b) Retention = fraction of the *previous* memory's bullets that reappear in the new memory —
exact string match, and "fuzzy" = token-Jaccard >= 0.6.

## 1. Quantitative table

| Metric | early 108–125 | mid 140–160 | late 175–190 | Reading |
|---|---|---|---|---|
| memory chars (mean±sd) | 1388 ± 396 | 1324 ± 351 | 1695 ± 382 | Memory grows only 22% end-to-end; it is not runaway inflation. |
| bullets / memory | 5.24 | 6.67 | **14.00** | Bullet count nearly triples — the visible behavioral change of the run. |
| chars / bullet | 266 | 200 | **123** | Bullets get shorter faster than they multiply: prose sentences fragment into terse notes. |
| bullet retention, exact | 0.0% | 0.0% | 5.2% | Verbatim carry-over is essentially absent until very late. |
| bullet retention, fuzzy | 0.4% | 1.5% | **12.9%** | Real but small; onset is sharp at rollout ~167 (0.5% → 7% → 13%). |
| `format_ok` (trained) | 39.0% | 27.7% | 42.6% | No trend, huge rollout-to-rollout swings (10.9% at r155, 70.3% at r125). |
| `format_ok` if the submit-string rule were dropped | 96.8% | 99.0% | **99.4%** | The structural 2-section format is *solved*; the gate is a single-string lottery. |
| memories containing `COMPLETE_TASK_AND_SUBMIT` | 47.3% | 56.9% | 49.2% | Half of all memories self-destruct on the `_PROMPT_MARKERS` check. |
| write_reward, all trained | 0.068 ± 0.211 | 0.047 ± 0.182 | 0.088 ± 0.230 | 65% of trained writes get exactly 0.0 (gate closed). |
| write_reward given `format_ok` | 0.175 ± 0.309 | 0.170 ± 0.315 | 0.207 ± 0.316 | Once the gate opens the reward is a noisy ±0.3 downstream delta. |
| `</think>` closure rate | 99.48% | 99.78% | 99.80% | Closure *improves*; no drift toward the sibling run's failure. |
| `finish == "length"` | 0.06% | 0.07% | 0.13% | Truncation is not a factor (45 unclosed cases in 15,936 total). |
| raw output chars / response tokens | 3466 / 749 | 3120 / 699 | 3324 / 798 | Thinking budget is stable; the extra bullets come out of the monologue, not extra tokens. |
| pure prose (no `###` header at all) | 0.52% | 0.10% | 0.20% | Format collapse never happens. |
| `explore_score` (judge, coverage 100%) | 0.721 | 0.724 | 0.690 | The independent novelty judge sees **no** improvement — slight decline. |
| `- Repository: ...` as lead bullet | 2.7% | 5.0% | **93.7%** | A rigid template crystallizes late. |
| stray `</memory>` tag inside memory | 0.1% | 0.0% | 9.0% | Late policy invents an XML wrapper and leaks the closing tag. |

## 2. Three memories, side by side

**Early — r112, `deepdiff`, `format_ok=False`, reward 0.0** (trimmed)

```
### Repository Knowledge
- The bug was in `deepdiff/operator.py` within `BaseOperator.match()`, which had inverted logic
  (`return False` instead of `return True`) causing custom operators with type matching to fail...
- Submission failed due to failed tests (`tests_failed`), indicating that the existing test suite
  should be run via `python -m pytest` or similar before submission...

### Lessons & Pitfalls
- Always run the full test suite before submission to catch unrelated failing tests...
- The submission command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` requires all tests to pass...
- Use a quick inline Python verification script to confirm the specific fix works before submitting...
```
Five long, essay-like sentences about *this one* bug plus generic advice. Nothing here is reusable
on the next issue of the same repo: no directory layout, no module map, no test invocation that is
specific to `deepdiff`. It also writes the submit command verbatim, which trips `_PROMPT_MARKERS`
and zeroes the reward — the single most common self-inflicted failure in the whole run.

**Mid — r150, `pandas`, `format_ok=False`, reward 0.0** (trimmed)

```
### Repository Knowledge
- Fixed a `NameError`/`UnboundLocalError` in `pandas/core/indexes/datetimes.py` where
  `indexer_between_time()` referenced `start_micros`/`end_micros` before definitions...
- Corrected docstring placement in `indexer_between_time()`...
- Validated the fix by importing pandas and executing the reproduction case...

### Lessons & Pitfalls
- In Python, variables must be defined before use within a function; `UnboundLocalError` triggers
  when a name is referenced before assignment...
- Complex multi-variable reordering is reliably applied via a single heredoc-based multi-line
  replacement rather than chained `sed` edits...
- Submit the completed patch with exactly `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`...
```
Same shape, one more bullet, and a shift toward *tool* lessons (heredoc vs `sed`) — genuinely
transferable, but transferable across every repo, not knowledge about pandas. What is dropped is
still everything from the previous five issues: this memory names one file and one function. The
"in Python, variables must be defined before use" bullet is the tell — the model is spending its
memory budget re-deriving language semantics.

**Late — r180, `sunpy`, `format_ok=True`, reward 0.717** (trimmed to 14 of 16 lines)

```
### Repository Knowledge
- Repository: `swesmith/sunpy__sunpy` (issue ID: `f8edfd5c`)
- Issue 1: `hgc_to_hgs` coordinate transformation broken with `UnboundLocalError`...
- Issue 2: `RotatedSunFrame` coordinate transformation hangs indefinitely due to broken
  `reference_to_rotatedsun` function in `sunpy/coordinates/metaframes.py`...
- Root cause for Issue 2: malformed `reference_to_rotatedsun` with return statement before
  variable definitions...
- Key modules: `sunpy/coordinates/metaframes.py`, `sunpy/coordinates/_transformations.py`,
  `sunpy/coordinates/frames.py`

### Lessons & Pitfalls
- Transformation graph decorated functions (`@frame_transform_graph.transform`) require strict
  variable ordering; never use variables before definition
- Use heredoc (`<<'EOF'`) for multi-line function replacements...
- Pattern reference: functions like `hgs_to_hgc`, `hcc_to_hpc`, `hpc_to_hcc` provide correct
  transformation structure templates
- Submission timed out due to extensive test suite execution (300s timeout on pytest)...
</memory>
```
This is the late style: a `Repository:` header bullet (93.7% of late memories), an enumerated
per-issue changelog, a "Key modules" bullet, and repo-specific *pattern references* — the first
genuinely reusable artifact in the run. It is also the only window where a stray `</memory>` tag
leaks (9%). What is still dropped: everything about issue 1 except one line, and any statement of
what the repo *is* beyond its name.

## 3. Does memory accumulate?

No — with one cosmetic exception late in training. Aggregated over full 12-issue episodes
(issues 1–6 = repo A, 7–12 = repo B):

| | early | mid | late |
|---|---|---|---|
| bullets of memory-after-issue-1 surviving into memory-after-issue-6 (same repo) | 0.0% | 0.0% | 5.8% |
| surviving into memory-after-issue-12 (across the repo switch) | 0.0% | 0.0% | 0.3% |
| memory-after-issue-6 → memory-after-issue-12 (cross-repo) | 0.0% | 0.0% | 0.6% |
| bullet count, issue 6 / issue 1 | 1.09 | 1.08 | 1.29 |

Concrete cases. **r112 `ep_1806`, sqlglot** (rewards `.88 .75 0 0 0 0 …`): after issue 1 the memory
is entirely about `extract_type` in `optimizer/simplify.py`; after issue 6 it is entirely about
`_add_date_sql` in `dialects/hive.py`. Retention 0.00 — `simplify.py` is gone, and the memory is
one issue deep at all times. **r150 `ep_2414`, astroid**: issue 1 records
`_forbid_class_getitem_access` in `brain_typing.py`; issue 6 records `_is_property` in `bases.py`.
Retention 0.00; not even "astroid is a static-analysis library" survives. **r182 `ep_2927`,
jinja** (the late style, retention 0.17): the memory after issue 6 opens
`Repository: swesmith/pallets__jinja - Jinja2 template engine, Issue 6/12` — the model now tracks
*where it is in the episode* — but the body is 100% `debug.py:get_template_locals`; the `do_default`
work from issue 1 has vanished.

That 0.17 is the ceiling, and it is boilerplate. Of the 4,854 bullets retained across consecutive
memories in the late window, **52% match `Repository:` / `Submission command:`** — the top retained
strings are literally ``Submission command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` `` (87x)
and ``Repository: `swesmith/stanfordnlp__dspy...` `` (66x). The only thing the model learned to
carry forward is the header line and the one string that closes its own format gate.

## 4. Is the reward teaching anything?

Distribution over 10,226 trained writes: mean 0.063, sd 0.206; **64.8% are exactly 0.0** (gate
closed), 26.3% > 0, 8.9% < 0; quantiles p5 −0.183, p25 0.000, p50 0.000, p75 0.067, p95 0.542.

Correlations of `write_reward` with candidate memory-content features, over the 3,597
gate-passing trained writes:

| feature | corr |
|---|---|
| memory chars | −0.001 |
| bullets | +0.039 |
| fuzzy retention | +0.034 |
| `explore_score` (LLM novelty judge) | −0.051 |
| **downstream window delta** `mean(r[k+1..k+3]) − mean(r[k−2..k])` | **+1.000** |
| next-issue minus current-issue reward | +0.557 |

Within the GRPO group where the advantage is actually formed (same episode, same rewrite index,
8 rollouts, 672 demeaned pairs) the content correlations vanish entirely: chars −0.056,
bullets −0.024, explore_score **+0.009**.

Plainly: **the WRITE reward is a measurement of the ACT policy's luck on the next three issues,
not of the memory.** By construction it equals the downstream window delta; empirically it carries
no information about anything the WRITE head controls. The only channel from memory content to
reward is the binary format gate — and 99.4% of late memories already satisfy every structural
clause, so that gate degenerates into "did the model happen to quote the submit command this
time", which oscillates between 20% and 83% per rollout with no trend. The bullet-count and
retention changes in section 1 are therefore best read as drift of the shared policy (pulled by the
ACT objective and by whatever the model finds natural), not as reward-driven learning; the
independent `explore_score` judge, which is the only content-sensitive measurement here, went
0.721 → 0.690.

## 5. The failure mode to watch: unclosed `</think>`

The rollout code takes the text after `</think>` as the new memory; if the WRITE output contains no
`</think>` at all it falls through neither branch of the guard at `codebase_rollout.py:1050` and the
**entire raw monologue becomes the memory** (this is what killed the non-thinking sibling around
rollout 155). In this run it stayed marginal: **45 unclosed outputs out of 15,936 (0.28%)**, and it
did **not** grow — closure was 99.48% early, 99.78% mid, 99.80% late, with the worst single rollout
at 98.44% (r128) and 26 of the 83 rollouts perfect. `finish == "length"` is 0.06% → 0.13%, i.e. the
handful of unclosed cases are mostly not truncation. The downstream symptom, a memory with no
`###` header at all, tracks it exactly: 0.52% → 0.10% → 0.20%. Verdict: the failure mode is present
but static; it is a monitoring item, not the reason this run's memory fails to accumulate.
