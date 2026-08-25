# ACT-behavior case study — `smith-4b-v3nocurr-explore-gwin-think`

Qwen3.5-4B SWE agent, ACT+think mode (`<think>` + one ```bash block per turn), 40-turn budget per issue,
episode = 2 repos x 6 issues with memory carried across issues. Run finished at rollout 190.

**Data.** `/project/flame/qixinx/backups/smith-4b-v3nocurr-explore-gwin-think-resume/traj/train/rollout_N/`,
N in {108..125, 140..160, 175..190} = 55 rollouts x 16 episodes x 12 trials = **10,560 trials / 179,905 turns**.
Every turn is read from `trials[].turns[]` (`assistant`, `assistant_think`, `action.command`, `observation`).

**Reward is exactly** `success x (1 - 0.025 x (turns - 1))`, verified on all successful trials (failure = 0.0).
One extra turn costs 0.025; a failed submit costs the whole ~0.8. This is the pressure the policy is responding to.

### Metric definitions (applied identically in all three windows)

- **Command buckets** — each turn's command is bucketed by the head of its first non-comment line, with write-detection
  overriding: `edit` = anything that writes a file (heredoc `> file`, `sed -i`, `patch`, `git checkout/apply`, an inline
  python script whose body opens a file for writing — this includes writing a `/tmp/fix_*.py` patch script);
  `test` = executing something without writing (`pytest`, `unittest`, `python x.py`, `python -c`, `./script`);
  `search` = `grep|find|ls|rg`; `view` = `cat|nl|head|tail|sed -n|wc|git diff`; `submit` = the turn whose observation is
  `Submission PASSED/FAILED`; `empty` = no command parsed.
- A turn like `pytest && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` is **not** a submit — the env ignores the token when
  combined; such turns are bucketed by the remaining command and counted separately as *combined-submit attempts*.
- **tested-before-submit** = among submitting trials, fraction with >=1 `test` turn anywhere before the submit.
- **blind-submit** = among submitting trials that contain >=1 `edit`, fraction with **no** `test` turn after the *last* edit.
- **prose chars/turn** = visible assistant text with fenced code blocks removed (the "visible chars" column still contains them).

---

## 1. Quantitative evolution

| Metric | early (108-125) | mid (140-160) | late (175-190) | one-line reading |
|---|---|---|---|---|
| trials / rollouts | 3456 / 18 | 4032 / 21 | 3072 / 16 | 2 GRPO groups per rollout, 24 fresh instances each |
| success rate | 0.444 | 0.488 | 0.466 | rises then flattens; late is not better than mid |
| mean reward | 0.318 | 0.369 | 0.354 | reward gain comes mostly from the step term, not from more solves |
| turns / trial | 17.97 | 17.68 | **15.15** | -16% turns: the clearest, most monotone behavioral gain |
| visible chars / turn | 875 | 944 | 1018 | grows only because the command block grows |
| prose chars / turn (code stripped) | 269 | 206 | 225 | visible prose shrinks ~20%: the answer becomes a thin wrapper on the command |
| think chars / turn | 1090 | 1091 | 1153 | thinking never shortens; reasoning is where the work migrated |
| mix: search | 17.3% | 17.8% | 17.7% | flat |
| mix: view | 22.5% | 18.8% | 20.1% | slight drop |
| mix: edit | 24.6% | 27.8% | **28.5%** | edits take a larger share of a shorter trial |
| mix: test | 28.9% | 29.1% | **25.9%** | the only bucket that shrinks |
| mix: submit | 4.6% | 4.6% | 5.8% | mechanical consequence of shorter trials |
| mix: empty | 1.9% | 1.6% | 2.0% | no format decay, no format improvement |
| tests / trial (absolute) | 5.18 | 5.14 | **3.92** | -24% executions per trial |
| edits / trial (absolute) | 4.42 | 4.92 | 4.32 | edit count flat; edits get bigger, not more numerous |
| **tested-before-submit** | **0.891** | 0.735 | **0.625** | verification before submitting decays steadily |
| **blind-submit** | **0.147** | 0.304 | **0.419** | nearly 3x: the dominant behavioral change of the run |
| last action before submit = `edit` | 7.3% | 26.6% | 36.7% | "patch then immediately submit" becomes a standard ending |
| last action before submit = `test` | 77.5% | 63.9% | 52.8% | mirror image |
| 40-turn cap-hit rate | 18.7% | 20.3% | **15.9%** | fewer dead-end thrashes (cap-hit success is 0.01) |
| empty-command rate | 1.94% | 1.63% | 2.02% | stable |
| multi-bash-block rate | 0.06% | 0.03% | 0.05% | format constraint held all run |
| placeholder echo ("Your reasoning and analysis here.") | 0.00% | 0.13% | 0.03% | never a real failure mode here |
| mean edit-command length (chars) | 1958 | 2269 | 2448 | +25%: bigger single-shot patches |
| heredoc share of edits | 82.7% | 90.1% | 92.8% | heredoc becomes the near-exclusive edit tool |
| `sed -i` share of edits | 8.0% | 2.3% | **1.1%** | line-surgery is abandoned (it silently mis-edits) |
| combined-submit attempts (`... && echo SUBMIT`, ignored by env) | 0.74% | 1.73% | 2.92% | 4x: an emerging wasted-turn tic |
| "output too long" warnings / turn | 5.1% | 6.4% | 8.0% | localization got *less* targeted, not more |
| within-window success: blind vs tested submits | 0.350 / 0.586 | 0.436 / 0.673 | 0.393 / **0.684** | the abandoned behavior remains the higher-scoring one |

---

## 2. Three representative trials

### Early — rollout 108, `graphql-python__graphene...combine_file__k1zx9xj1` (13 turns, reward 0.70)
Dominant early shape: edit -> run tests -> read failure -> re-edit, repeated until green.

```
T1  [search] find . -name "*.py" | xargs grep -l "SimpleGlobalIDType\|UUIDGlobalIDType" -> 3 files incl. tests/test_custom_global_id.py
T2  [view  ] cat ./graphene/relay/id_type.py                                  -> full source of both ID classes
T3  [view  ] cat ./graphene/relay/tests/test_custom_global_id.py              -> WARNING: output too long
T4  [edit  ] sed -i 's/_type, global_id/.../g' ./graphene/relay/id_type.py    -> rc=0, no output
T5  [test  ] cat id_type.py && python -m pytest .../test_custom_global_id.py  -> rc=1, tests still failing
T6  [edit  ] cat <<'EOF' > /tmp/fix_id_type.py ... (rewrite resolve_global_id) -> "File updated successfully"
T7  [test  ] python -m pytest .../test_custom_global_id.py -v                 -> rc=1, new error: Relay Node "None" not found
T8  [view  ] cat ./graphene/relay/id_type.py                                  -> current post-patch source
T9  [edit  ] cat <<'EOF' > /tmp/fix_id_type.py ... (fix UUIDGlobalIDType)      -> "File updated successfully"
T10 [test  ] python -m pytest .../test_custom_global_id.py -v                 -> rc=1, UUID tests pass, Simple still fails
T11 [edit  ] cat <<'EOF' > /tmp/fix_id_type.py ... (fix to_global_id prefix)   -> "File updated successfully"
T12 [test  ] python -m pytest .../test_custom_global_id.py -v                 -> rc=0, 8 passed
T13 [submit] echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT                       -> Submission PASSED
```

Three independent bugs are found *only* because the test suite is re-run after each patch; each new error message
becomes the next hypothesis. The reasoning is explicitly observation-driven, quoting concrete failure text rather than
the model's own prediction. The cost is 13 turns (reward 0.70) versus the ~0.95 a 3-turn solve would earn — exactly the
gradient that later erodes this loop. Note T4: an unverified `sed -i` produced no visible effect, and only the T5 test
run exposed it.

> "The sed commands ran but didn't output anything visible (returncode 0 means success). However, I should verify the changes were applied correctly by viewing the file again, and then run the tests"

> "The error shows that `result.data["user"]["id"]` is `'User:my global primary key in clear 3'` instead of `'my global primary key in clear 3'`."

> "Good progress! Now the `TestUUIDGlobalID` tests pass, but `TestSimpleGlobalID` still fails."

### Mid — rollout 141, `tweepy__tweepy...func_basic__fbrrylvw` (7 turns, reward 0.85)
Dominant mid shape: whole-file heredoc rewrite, one cheap test probe, then *read the test file instead of re-running it*.

```
T1  [search] grep -r "list_to_csv" --include="*.py" .            -> hits in tweepy/api.py, tweepy/utils.py
T2  [view  ] cat tweepy/utils.py                                 -> buggy `', '.join(map(str, item_list[:-1]))`
T3  [edit  ] cat <<'EOF' > tweepy/utils.py  (whole file rewritten) -> rc=0
T4  [test  ] pytest                                              -> rc=1, WARNING: output too long
T5  [view  ] cat tests/test_utils.py && cat tweepy/utils.py      -> expected "1,2,3"; current code emits "1, 2, 3"
T6  [edit  ] cat <<'EOF' > tweepy/utils.py  (separator ', ' -> ',') -> rc=0
T7  [submit] echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT          -> Submission PASSED
```

The single `pytest` at T4 is uninformative (truncated), and instead of narrowing it (`pytest tests/test_utils.py -q`) the
policy switches to *reading the assertion* and reasoning out the expected string. That substitution works here — the
second bug (separator spacing) is caught — but the final patch is never executed. This is the mid-window bargain in
miniature: an oracle-quality read replaces a run, saving 3-4 turns (+0.10 reward) at the price of an unchecked last edit.

> "The tests show that there are some failures... Let me check the current state of `tweepy/utils.py` to see if my fix was applied correctly."

> "The test expects: `list_to_csv([1, 2, 3])` -> `"1,2,3"` ... The current implementation with `', '` would produce `"1, 2, 3"` (with spaces)"

> "The fix has been applied. Now I need to submit the final output to complete the task."

### Late — rollout 177, `cantools__cantools...func_pm_ctrl_invert_if__wl0rywhr` (6 turns, reward 0.875)
Dominant late shape (search->view->edit->submit is the last-4 pattern of 17.8% of late submitting trials, up from 1.6% early): locate, read one window, one large
heredoc patch, submit — with the verification carried out *inside the think block* rather than in the environment.

```
T1  [search] grep -r "subplot" --include="*.py" .                        -> WARNING: output too long
T2  [view  ] cat ./src/cantools/subparsers/plot.py                       -> WARNING: output too long
T3  [search] grep -n "subplot" ./src/cantools/subparsers/plot.py         -> line numbers for SEP_SUBPLOT / SEP_AXES logic
T4  [view  ] nl -ba ./src/cantools/subparsers/plot.py | sed -n '500,680p' -> the __init__ separator loop
T5  [edit  ] cat <<'EOF' > /tmp/fix_plot.py  (swap SEP_SUBPLOT/SEP_AXES branch) -> "Fix applied successfully"
T6  [submit] echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT                  -> Submission PASSED
```

Two turns are burned on over-broad reads (T1, T2) that return only a truncation warning — the "too long" rate is highest
late (8.0%/turn), so localization did not get more surgical, it got more tolerant of waste. The decisive step is T5's
think block, which *hand-simulates* the loop on a synthetic input and derives the inverted condition. The only feedback
before submitting is the fix script's own `print("Fix applied successfully")` — a string the model wrote itself. Late
pre-submit observations are under 150 chars 36.1% of the time (vs 18.2% early): the last thing the policy sees before
submitting is increasingly its own echo, not a test result. Here the mental simulation was right; in the 60.7% of late
blind submits that fail, it was not.

> "Let me trace through an example: Example: `signal_group_1 - signal_group_2` ... `self.subplot_axis += 1` (axis becomes 1, not subplot)"

> "Original: checked if `signals[i1] == self.SEP_SUBPLOT` to increment the axis - Fixed: check if `signals[i1] == self.SEP_AXES`"

> "The fix has been applied... Now let me submit the solution."

---

## 3. What the policy actually learned

**(a) Turn efficiency, bought mostly from verification.** Turns/trial fell 17.97 -> 15.15 (-16%) and cap-hits 18.7% ->
15.9%, while edits/trial stayed flat (4.42 -> 4.32) and tests/trial fell 5.18 -> 3.92 (-24%). The step budget was
recovered almost entirely from the execute-and-check phase, not from faster localization.

**(b) Verification collapse / blind submission.** tested-before-submit 0.891 -> 0.625; blind-submit 0.147 -> 0.419;
last-action-before-submit = `edit` 7.3% -> 36.7%; pre-submit observations shorter than 150 chars 18.2% -> 36.1%. The
policy learned to treat "the patch script printed success" as sufficient evidence. Within every window the abandoned
behavior scores better (late: 0.684 tested vs 0.393 blind), so this is a reward-shaping artifact, not a discovered
improvement — it is cheap (each verification turn costs 0.025) and its cost only shows up as a rare catastrophic zero.

**(c) Edit style consolidated on whole-block heredoc rewrites.** `sed -i` share of edits 8.0% -> 1.1%, heredoc share
82.7% -> 92.8%, mean edit-command length 1958 -> 2448 chars. The early trace shows why: a `sed -i` that silently matched
nothing (T4) was only caught by a test run. With tests being dropped, the policy migrated to the edit primitive that is
self-reporting and idempotent — a coherent adaptation to its own reduced feedback.

**(d) Reasoning absorbed the work that the environment used to do.** Visible prose per turn fell 269 -> 225 chars while
think length rose 1090 -> 1153, and the late exemplar replaces a test run with an in-`<think>` hand-simulation of the
buggy loop. Localization itself did not improve: search share is flat (17.3% / 17.8% / 17.7%), turn-of-first-edit is flat
(5.68 / 5.72 / 5.48) and truncation warnings rose 5.1% -> 8.0%/turn. Format compliance was never the issue — multi-bash
0.06% -> 0.05%, placeholder echo <=0.13%, empty commands ~2% throughout — but one new tic appeared: combined
`... && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` turns, which the env silently ignores, grew 4x (0.74% -> 2.92%).

---

## 4. Caveats

Per-rollout success ranges 0.21-0.72 within a single window (early [0.26, 0.67], late [0.21, 0.67]), so the window-level
success differences (0.444 / 0.488 / 0.466) are well inside noise and I do not claim the policy got better at solving —
only that its *shape* changed, and the shape changes (blind-submit +0.27, tests/trial -24%, turns -16%) are large
relative to that spread and monotone across all three windows. Each rollout contains only 2 GRPO groups (24 distinct
instances), and instance sets are disjoint across windows (zero shared instance ids; 20 of 28/34 repos overlap), so
difficulty is a real confound: an easier draw simultaneously raises success, shortens trials and reduces the need to
test, which would manufacture part of the same correlation. The blind-vs-tested success gap is observational, not
causal — a model that is confident because the bug is a one-line inversion both skips the test and succeeds, so the
0.684/0.393 gap is an upper bound on the cost of skipping verification. Bucketing is heuristic: writing a `/tmp/fix_*.py`
patch script counts as `edit` while running it counts as `test`, and a `python -c` repro is counted as `test` even when
it is really exploration. Finally, the memory channel means trial position confounds within-episode trends (issues 7-9,
the first of the second repo, run at 0.60-0.65 success late vs 0.40-0.45 for issues 3-6); all numbers above pool the 12
positions equally, which is fair across windows but hides that effect.
