# gwin1 (K=1) r60–80: ACT reward decline + write_overlong spike — trajectory analysis (2026-08-19)

Run: `smith-4b-v3nocurr-explore-gwin1` (K=1 windowed WRITE reward, seq_length=24576).
Data: `/project/flame/qixinx/backups/smith-4b-v3nocurr-explore-gwin1/traj/train/` (rollouts 0–125, full).
Three parallel trajectory-analysis agents (ACT side / WRITE side / trigger+recovery); this file is the
merged synthesis plus each agent's key evidence.

## Merged verdict (causal chain)

1. **Root cause: gradual ACT policy drift into a degenerate whole-file-rewrite loop**, onset ~r56–63
   (ramp, not a step), peak r80–82. The policy shifted from "targeted sed → submit" (median 15 turns,
   30k chars/trial) to emitting 5–10k-char `cat <<'EOF' > file.py` full-file rewrites and long
   command-less monologues, repeated near-verbatim because the empty `<returncode>0</returncode>`
   observation gives no corrective signal. Turns/trial 15.1→26.6, assistant chars/turn 878→1472,
   heredoc-rewrite turns 10–16%→40% (r69). Extreme case: 37 consecutive full-file rewrites in one
   trial (r69 dspy, 340k chars ≈ 4× seq_length).
2. **Reward fell via budget exhaustion, not worse diagnoses**: 40-turn cap-hit 16.5%→53.6%, submit
   share of turns 5.8%→0.6%; success 0.29→0.15. Repo-mix confound excluded (all 23 shared repos show
   the same inflation; K=3 sibling on same schedule didn't spike). iter71 crash/restart: no
   discontinuity (r72 slightly *better* than r70–71).
3. **act_overlong (0.07→0.72) is purely transcript ballooning** (turns × assistant verbosity;
   observations shrank, memory_in flat ~2.8k chars). A chars>88k proxy reproduces the wandb curve
   point-by-point.
4. **write_overlong spike (0.03→0.17) is 100% prompt-side collateral**: WRITE prompts embed the full
   trial transcript; response length, memory length, format_ok, thinking all stayed flat. Median WRITE
   input 32.5k→93.7k chars. WRITE generation itself stayed healthy — memories remained clean
   two-section bullet lists even with 373k-char prompts; no narration creep.
5. **write_mean going negative (r70–115) indicts the reward plumbing, not the memories**:
   - Verified exactly: `write_reward == reward[k+1] − reward[k] + 0.1` (corr 1.0, residual 0);
     per-sample std ~0.4 = instance-difficulty sequencing noise swamps any content signal (same
     episode, two equally good adjacent writes: +1.05 then −0.85).
   - **Selection bias from overlong-dropping**: prompt-overlong WRITE samples are dropped from
     training (`trained=False`), and they are overwhelmingly post-failure samples (cur≈0, delta
     positive +0.06..+0.13). Censoring them turns a ~0 unconditional delta into −0.07..−0.10 on the
     trained set. Trained fraction collapsed 157–171/192 → 26–41/192 (r80–81).
   - Audit-trained mean (+0.02..+0.03) vs wandb (−0.006..−0.064) gap: the agent inferred a ~−0.5
     truncation override from a two-point solve, but **no such penalty exists in code** (tail-trim
     keeps reward; explicit comment). wandb `write_mean` is computed post zero-std filter
     (`log_rollout_data` runs after `drop_zero_std_groups`), so the two means are over different
     sample sets; exact reconciliation left open, but no hidden penalty constant.
6. **Self-recovery (r83+) was gradient selection through truncation**: overlong tail-truncation
   removes gradient from late-trial policy tokens, while terse trajectories that fit 24576 keep full
   signal → surviving gradient favored compression. Thought chars/turn 1345→177 (r94+), heredoc turns
   27%→7%, empty commands 8%→1%, submit back to 50–70%, reward back to ~0.18. But turns stayed ~30
   (cap-hit ~50%): the policy traded rewrite loops for slow view-heavy exploration, never re-learning
   the early fast edit-and-submit path.
7. **r120–125 tail collapse is a different disease**: transcripts short (25–29k), overlong ~0.015,
   but empty-command rate 3.4%→81% with "OK OK OK…" babble (same OK-spam format collapse as
   smith-v2 late phase); ramp visible from r116.

## Practical implications

- The K=1 (and family) WRITE reward under long-transcript conditions trains on a shrunken
  (14–21%), difficulty-sequencing-biased sample with near-zero content information.
- Overlong drop/truncation is not neutral plumbing: it censors post-failure WRITEs (biasing the
  delta negative) and deletes late-trial ACT gradient (which both fed the sickness and drove the
  recovery). Any future arm with long transcripts inherits both effects.
- The whole-file-rewrite loop is the same attractor documented in the K=3 case study (py-replace
  whole-function rewrites) taken to its extreme; "empty observation after a successful write" is the
  missing corrective signal.
- Placeholder-sentence leakage ("Your reasoning and analysis here.") also appeared in K=1 sick-window
  thoughts (r75 sqlglot case).

## Key evidence tables

ACT window means (n=trials):

| window | n | reward | succ | turns | asst ch/turn | obs ch/turn | mem_in | tot ch p50/p90 | over-88k | empty-cmd | submit share | cap-40 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r40–55 | 3072 | 0.197 | 0.257 | 15.1 | 878 | 1137 | 2794 | 30.5k/89.9k | 0.11 | 2.6% | 5.8% | 16.5% |
| r60–80 | 4032 | 0.119 | 0.201 | 26.6 | 1472 | 962 | 2768 | 73.1k/137.9k | 0.39 | 4.4% | 2.3% | 53.6% |
| r85–100 | 3072 | 0.111 | 0.196 | 29.9 | 681 | 1472 | 2522 | 75.3k/119.5k | 0.37 | 2.4% | 1.5% | 49.9% |

WRITE window means:

| window | input ch p50 | prev-mem | raw out | format_ok | finish=len | est overlong |
|---|---|---|---|---|---|---|
| r40–55 | 32.5k | 2794 | 5810 | 0.959 | 3.3% | 0.098 |
| r70–80 | 93.7k | 2805 | 5914 | 0.987 | 1.8% | 0.501 |
| r85–95 | 79.9k | 2629 | 6165 | 0.977 | 8.1% | 0.377 |

Delta decomposition (K=1 pairs):

| window | ALL Δ(next−cur) | TRAINED Δ | DROPPED Δ (cur/next) |
|---|---|---|---|
| r40–55 | −0.013 | −0.032 | +0.129 (0.011/0.140) |
| r70–80 | −0.003 | −0.070 | +0.060 (0.019/0.079) |
| r85–95 | −0.018 | −0.098 | +0.070 (0.008/0.078) |

Same-repo transcript chars pre/sick/post: dspy 27.7k→96.5k→52.4k (assistant 14.5k→79.7k→19.3k);
sunpy 34.9k→91.5k→59.9k; oauthlib 23.3k→70.1k→52.4k; sqlfluff 41.4k→71.6k→52.6k;
cantools 46.5k→63.0k→55.6k; pandas 46.9k→78.0k→69.9k.

Onset per-rollout (assistant ch/turn): r59 892 → r63 1303 → r69 2277 → r75 1703 → r80–82
1420–1640 (turns 36.5–37.1, submit 0.5%) → r85 ~590 → r94–99 ~500.

## Representative cases

- r69 dspy `pr_7914`: 37 consecutive ~8k-char full-file rewrites of `parallelizer.py`, zero tests,
  zero submit, 340k chars.
- r75 sqlglot `combine_module__3uqs5jcs`: correct diagnosis by t4, then ~20 near-identical full-file
  rewrites of prql.py; t32 thought begins with the prompt scaffold "Your reasoning and analysis
  here." Never submits.
- r75 click `func_basic__67b198bc`: writes reproduce.py at t1, never runs it; ~35 rewrites; 10k-char
  thoughts ending in empty commands.
- r75 cantools (success, rew 0.225): correct one-line fix at t1, then 17 turns of 10k-char thoughts
  and rewrites, git checkout to undo its own mess, resubmits at t32 — success still overlong.
- WRITE C1 (r73 dspy, input 373k chars, dropped): output correctly reasons "my previous memory was
  about a COMPLETELY DIFFERENT task" and produces a clean 1.5k-char two-section memory — nothing
  wrong with the WRITE; it was simply never trained.
- WRITE C3/C4 (r72 sqlfluff, same episode): +1.05 then −0.85 for two adjacent, equally sound
  writes — pure sequencing noise.
- r123 ACT turn (late collapse): `"OK OK OK. Ok. ![image](https://i.imgur.com/…) … OK OK OK"` with
  empty command, 81% empty-command rate at r125.

Analysis scratch data (per-trial stats.jsonl / trials.jsonl, sweep scripts) were in the session
scratchpad (volatile); traj source is durable on /project.
