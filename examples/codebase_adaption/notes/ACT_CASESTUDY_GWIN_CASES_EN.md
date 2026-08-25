# Case Studies: ACT Behavior Early vs. Late in Training — `smith-4b-v3nocurr-explore-gwin`

English translation of Section 2 ("Case study") of the ACT-evolution trajectory case study.
Case pairs are matched on **same repo and same mutation type**; instances never repeat during
training, so no pair shares the exact same `instance_id`.

Excerpt format: `Tn [bucket] command → one-line observation`. The full excerpts were in
`act_excerpts.txt` (lost with the node; the excerpts below are the curated subset).

---

## Case 1 — cantools `func_basic`: rollout 0 `func_basic__fnvf3wq5` (SYM `_dump_signals`) vs. rollout 124 `func_basic__xqus9738` (C header guard + fuzzer makefile)

**Early (rollout 0, 3 of 8 rollouts succeed, mean 38.4 turns; below is one that only PASSes after 40 turns)**

```
T01 [search] ls -la /testbed                                    → directory listing
T02 [search] find /testbed -name "sym.py"                        → ./src/cantools/database/can/formats/sym.py
T03 [view  ] cat …/sym.py                                        → output too long
T04 [view  ] sed -n '/def _dump_signals/,/^def /p' … | head -100 → sees the function
T05 [test  ] python3 -c "import cantools; db=load_file(...)"      → rc=1 Traceback (file does not exist)
T06 [search] find /testbed -name "*.sym" | head                  → finds tests/files/sym/*.sym
T07 [test  ] python3 -c "db=load_file('tests/files/sym/send-6.0.sym'); print(dump)" → reproduces the bug
… 33 more turns (repeated sed -i / rerunning the repro script / pytest -k sym), T40 submit → PASSED (reward 0.025)
```

Another rollout (failed): a multi-line `sed -i '855,873s/.*/…'` triggers a bash syntax error
(rc=2 unexpected EOF); the model switches to a `/tmp/fix_sym.py` script, then burns the rest of
the 40-step budget cycling through pytest.

**Late (rollout 124, 8 of 8 rollouts fail, 4–8 turns)** — the 8 rollouts are nearly isomorphic:

```
T01 [search] find . -name "*.py" | xargs grep -l "c_source\|generate\|header" | head -5 → c_source.py etc.
T02 [view  ] cat -n ./src/cantools/database/can/c_source.py      → **output too long (never actually sees the code)**
T03 [edit  ] cat <<'EOF' > /tmp/fix_c_source.py … content.replace(
             "    include_guard = f'{database_name.lower()}_h'  # Changed to lowercase.", …) … EOF && python /tmp/fix_c_source.py
                                                                 → rc=0 (no output)
T04 [submit] echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT          → Submission FAILED (tests_failed)
```

Commentary:
- Early behavior is the "over-explore + verify repeatedly" type: a repro script appears as early
  as T5, test commands make up 26% of turns, but tool use is clumsy (multi-line `sed`
  replacements with syntax errors, `cat` of whole files getting truncated). Exhausting the
  40-step budget is the dominant failure mode (55% of rollout-0 trials hit the budget).
- Late behavior is the "blind edit + instant submit" type: T2's output is truncated and **the
  model never sees the source code**, yet at T3 it writes an "old code" string in the replace
  script complete with a `# Changed to lowercase.` comment — a guess at / hallucination of the
  SWE-smith injected-comment style. The `str.replace` has no guard, silently fails to match, and
  T4 submits anyway. The whole group fails 8/8 → GRPO advantage is all zeros on this task, so no
  gradient ever tells the policy this is wrong.
- Across the 8 rollouts of this one task, the first 3 command types are identical
  (search → view → py_replace → submit): within-group diversity has collapsed.

---

## Case 2 — sqlglot `func_pm_ctrl_shuffle`: rollout 0 `rq4xvhqf` (`_add_replace_columns` UnboundLocalError) vs. rollout 150 `48dpisag` (`alias_or_name` in `eliminate_qualify`)

**Early (rollout 0, pos 6, memory_in 2.1k chars; success, 20 turns, reward 0.525)**

```
T01 [search] find . -name "*.py" | xargs grep -l "REPLACE" | head -20       → several files
T02 [search] grep -n "def qualify_columns" sqlglot/optimizer/*.py            → qualify_columns.py:19
T03 [view  ] cat sqlglot/optimizer/qualify_columns.py                        → output too long
T04 [search] grep -n "_add_replace_columns" …                                → 640/646/744
T05 [view  ] sed -n '744,800p' …                                             → sees the function
T06 [script] cat <<'EOF' > /testbed/test_replace_bug.py …                    → (writes a repro)
T07 [test  ] python test_replace_bug.py                                      → rc=1 UnboundLocalError (repro confirmed)
T08 [view  ] sed -n '744,760p' …                                             → looks again
… T9–T19: sed -i edit → repro script goes green → edge-case script → pytest; T20 rm test file && submit → PASSED
```

The memory holds 2k characters, yet T1 still opens with `find | xargs grep` — the memory did not
change the exploration path.

**Late (rollout 150, pos 6, memory_in 1–3.4k chars; 8 of 8 succeed, 3 turns, reward 0.95)** —
all 8 rollouts share a verbatim-identical T1:

```
T01 [view  ] cat -n sqlglot/transforms.py | grep -A 100 "def eliminate_qualify"   → sees the function directly (filename taken from the issue text)
T02 [edit  ] cat <<'EOF' > /tmp/fix_qualify.py … old_func='''def eliminate_qualify…''' new_func=… content.replace … EOF && python /tmp/fix_qualify.py → Done
T03 [submit] echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT                           → PASSED
```

Commentary:
- This is the "ideal path" the policy has learned: locate (1 step, direct view instead of
  search) → whole-function replacement (1 step) → submit; no repro, no tests. For
  ctrl_shuffle-style tasks ("one look tells you the statement order is wrong"), reward rose from
  0.5 to 0.95 — entirely from the turns term.
- The price is that no verification step remains anywhere; within the group, all 8 rollouts share
  identical command types at T1–T3, differing only slightly in the `new_func` text inside the
  replace script (some move a `return`, some rewrite a helper). GRPO's within-group variance has
  been reduced to "did this particular whole-function rewrite happen to be correct".

---

## Case 3 — astroid `lm_rewrite`: rollout 6 `08v06c41` (typing.NamedTuple brain registration) vs. rollout 125 `p9cqd5n6` (`util.safe_infer` returning None for Uninferable)

**Early (rollout 6, pos 5, memory_in 2–4.5k; 8/8 fail, all 40 turns)**

```
T01 [search] find /testbed -name "brain_namedtuple*" && grep -r "typing.NamedTuple" … → brain_namedtuple_enum.py
T02 [view  ] cat …/brain_namedtuple_enum.py                 → output too long
T03 [view  ] sed -n '280,350p' …                            → fragment
T04 [view  ] sed -n '440,490p' …                            → fragment
T05 [view  ] sed -n '680,750p' …                            → fragment
T06 [view  ] sed -n '350,430p' …  T07 sed -n '358,410p'  T08 sed -n '420,480p' …
… 32 more turns; the final turn is still a [view] sed -n '590,610p' → Max steps (40) reached
```

Another rollout: T1 `find /astroid …` (hallucinated path, rc=1), T2 `find / && find / -name …`
(output explosion, truncated), only at T3 does `$PWD` locate the file — then it falls into the
same `sed -n` pagination loop. **Read-only budget exhaustion** is the typical early death for
`lm_rewrite` (in the 0–19 window, fail_budget_never_edited is 5.7% overall, but within the
lm_rewrite type, 40-turn failures are the majority).

**Late (rollout 125, pos 11, memory_in 0.7–2.5k; 7 of 8 succeed, 3–5 turns)**

```
T01 [view  ] cat -n astroid/util.py | grep -A 20 "def safe_infer"        → sees the function (path from the issue)
T02 [edit  ] cat <<'EOF' > /tmp/fix_safe_infer.py … content.replace(old_code,new_code) … EOF && python /tmp/fix_safe_infer.py → Done
T03 [submit] echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT                  → PASSED
```

The one failing rollout: T1 `find … -exec grep -l "def safe_infer\|Uninferable"` → T2 view →
T3 replace script (a different fix: only changes the return value in the isinstance branch) →
T4 submit → tests_failed. Another rollout glues `cat <<'EOF' > /tmp/test.py && cat /tmp/fix.py`
onto the heredoc header → the harness judges it "Empty command" (one step wasted), then finishes
normally in 3 steps.

Commentary: this task (single function, file named directly in the issue) happens to sit in the
late policy's comfort zone. Contrast with rollout 6's kind of lm_rewrite — "understand the
registration relationships among multiple transforms in a 700-line brain file" — which the late
policy handles by "change something arbitrary and submit" (window 120–139: lm_rewrite success
0.14 at 21.8 turns, vs. early 0.17 at 32.6 turns — same success rate, it just dies faster).

---

## Supplement: a mid-training sample (rollout 65, pdfminer `func_basic`, success, 10–12 turns)

```
T01 [search] find . -name "*.py" -exec grep -l "get_widths" {} \;    → pdffont.py
T02 [view  ] cat -n ./pdfminer/pdffont.py | grep -A 50 "def get_widths"
T03 [script] cat <<'EOF' > test_issue.py … (repro)
T04 [test  ] python test_issue.py                                    → repro: expected {32:250,…} got {33:250,…}
T05 [edit  ] cat <<'EOF' > fix.py … content.replace(...)              → (no output)
T06 [test  ] python test_issue.py                                     → still wrong (replace did not match)
T07 [view  ] nl -ba pdffont.py | sed -n '65,78p'                      → looks at the real text
T08 [edit  ] sed -i …  T09 test → green  T10 submit → PASSED
```

The 60–79 window is a brief "verification-habit rebound" (tested-before-submit 60%, blind submit
7–13%): there is a repro, and a regression run after the edit, yet the flow is already visibly
leaner than rollout 0 (10 steps vs. 20–40). After this window (rollout 100+), the verification
step gets squeezed out again by the reward.
