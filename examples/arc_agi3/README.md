# ARC-AGI-3 → Miles: memory-agent RL on Qwen3.5-4B

## STATUS (2026-06-09) — ✅ end-to-end smoke COMPLETE (SLURM job exit 0)

**The full M2 smoke ran to completion** on 4×L40S (job 8299361, `COMPLETED 0:0`): load → 4 sglang
engines → megatron init → **3 rollout+train cycles → save checkpoint (6 min) → clean exit 0**. Every
training step healthy (`ess_ratio≈0.998`, `train_rollout_logprob_abs_diff≈0.03`; `loss/grad=0.0` is
expected — sparse reward, 0/96 episodes completed a level → all-zero advantage). M1 (SGLang VLM +
logprobs) and the memory rollout (model writes coherent `<memory>`, reasons over the grid diff, parses
`<action>`; memory evolves turn-to-turn) both work.

**Winning config = 4×L40S, TP=4, `general` partition, sbatch.** The earlier "GPU memory" wall was a red
herring: with **TP=4** each rank holds only ~1B params, so the **fp32 Adam optimizer fits fully on-GPU
(~38/46 GB) — no offload, no bf16 hacks, clean config**. The real constraint was **CPU memory for the
colocate load phase** (sglang + megatron ranks load the 4B weights at once → ~49 GB peak): 48G/64G nodes
OOM at `actor_model.init()`; L40S `general` nodes have ~768G CPU (we request 256G) and never OOM.

**How to run (general, 4×L40S):**
```bash
sbatch examples/arc_agi3/sbatch_arc_qwen3.5_4B.sh
tail -f examples/arc_agi3/arc_smoke.log           # live log (symlink to /data file)
python3 examples/arc_agi3/read_traj.py --limit 5  # rollout trajectories (per-episode)
# NOTE: /data is mounted only on COMPUTE nodes, not login — view ckpt/log/traj from a compute node.
```

**To scale toward real training** (revert the smoke-size knobs — see memory `arc-mem-tweaks-revert`):
raise `--num-rollout`, `--rollout-batch-size`, `--n-samples-per-prompt`, `--global-batch-size`, and
`--rollout-max-response-len` (1024→2048+, ~26% of turns currently truncated mid-`<memory>`); consider a
larger `max_turns` so the model explores enough to occasionally complete a level (non-zero reward →
non-zero advantage → actual learning, instead of the all-zero-gradient smoke).

**Key fixes along the way** (run script / files here, plus the ARC-repo edit):
MTP-stripped checkpoint `/data/user_data/qixinx/qwen3.5-4B-nomtp/` (megatron.bridge can't load Qwen3.5's
dense MTP head); **TP=4**; `--qkv-format bshd --micro-batch-size 1` (Qwen3.5 GatedDeltaNet rejects packed
seqs); drop `mm_token_type_ids` from `multimodal_train_inputs` (variable length breaks the cat); fixed
64×64 render canvas (uniform pixel_values); RHEL→Debian `SSL_CERT_FILE` override (container CA bundle);
`MILES_DISABLE_MTP`/`PYTHONDONTWRITEBYTECODE`; ARC `arc_rl/arc_env.py` robust to SDK `None`/HTTP-500.
Image cell=14.

---

Train a Qwen3.5-4B VLM agent on [ARC-AGI-3](https://github.com/arcprize) interactive grid games with
GRPO, as a **memory-based agent**: each turn the policy sees only `[system + previous memory M_{t-1}
+ last transition + current grid image]` — never the past-turn history — and emits, in one
generation, an updated memory `M_t` then an action `a_t`. Design doc:
`/home/qixinx/ARC-AGI-3-Agents/docs/world_model_memory_rl.md`. Scope here = **M1 + M2** (plumbing +
GRPO smoke with the true sparse reward); the world-model auxiliary loss `L_WM` (M3) is **not** included.

## Files
| file | role |
|---|---|
| `prompts.py` | **all hand-written prompts**: `DEFAULT_SYSTEM` (rules/format) + `render_user_text(memory,last,obs)`. Edit prompt wording here. |
| `grid_render.py` | `render_grid(grid)->PIL.Image` (the image observation) + `grid_diff_text(a,b)` (the `d_t` perception input). Ported from ARC `memory_agent`. |
| `env_arc.py` | `parse_response(text)` (`<memory>`/`<action>` extraction, **stdlib-only, unit-testable offline**) + `ArcInteractionEnv` wrapping `arc_rl.arc_env.ArcEnv`. |
| `arc_rollout.py` | custom `generate` (`--custom-generate-function-path`): runs one episode, returns one `Sample` **per turn**; also writes a per-episode trajectory to `$ARC_TRAJ_DIR`. |
| `arc_advantage.py` | custom `--custom-reward-post-process-path`: **episode-level GRPO** whitening, broadcast to each episode's turn-samples. |
| `read_traj.py` | pretty-print the per-episode trajectories from `$ARC_TRAJ_DIR` (memory/action/reward per turn). |
| `arc_config.yaml` | `max_turns` (merged onto args via `--custom-config-path`). |
| `data/arc_games.jsonl` | one row per game instance (`prompt` placeholder + `metadata.game_id`). |
| `run_arc_qwen3.5_4B.sh` | 2-GPU (TP=2) colocated VLM GRPO launcher. |

## Logging (rollout trajectories + metrics)
- **Per-episode trajectory (readable, recommended):** `arc_rollout` dumps one `ep_<index>.json` per
  episode to `$ARC_TRAJ_DIR` (default `/data/user_data/qixinx/arc_traj`, set in the run script) — each
  turn's `memory` / `action` / `reward(Δlevels)` / `diff` / `finish`, **no images** so it stays tiny.
  Inspect with `python3 examples/arc_agi3/read_traj.py [--full] [--limit N]`. Disable by unsetting `ARC_TRAJ_DIR`.
- **miles multi-turn metrics:** `--log-multi-turn` (in the run script) logs response-length/round stats.
- **wandb:** enabled automatically when `WANDB_API_KEY` is in the env (the run script adds `--use-wandb
  --wandb-project --wandb-group --wandb-key`); override via `WANDB_PROJECT` / `WANDB_GROUP`. wandb x-axis
  is `rollout_id` (one point per `num_rollout` iteration). Without the key, wandb is off and metrics
  still print to stdout + the log.
- **miles full raw dump (heavy):** `--save-debug-rollout-data <path>/{rollout_id}.pt` dumps *every*
  sample incl. `pixel_values` (~hundreds of MB/rollout) — only enable for deep debugging.

## Why each turn is a plain single-turn sample
The per-turn context is self-contained (memory + current state, not concatenated history), so every
turn is an ordinary `(prompt, response)` RL sample. The multi-turn-ness lives only in (a) the rollout
loop that emits one sample per turn and (b) episode-level advantage grouping. No growing sequence,
no train/inference attention mismatch, exactly one image per turn-sample.

## Prerequisites
- **Miles runtime**: not on the host; run inside the apptainer image
  `/data/user_data/qixinx/images/miles_dev-202606081341.sif` (CUDA 12.x; the host driver 575/CUDA 12.9
  cannot run the CUDA-13 `:latest`).
- **Checkpoint**: `/data/hf_cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
  (confirmed VLM: `Qwen3_5ForConditionalGeneration`, vision+text config, image preprocessor).
- **ARC SDK in the rollout process**: `arc_agi`, `arcengine`, `python-dotenv`, `pillow`. `env_arc`
  adds `$ARC_AGI3_REPO` (default `/home/qixinx/ARC-AGI-3-Agents`) to `sys.path` for `arc_rl`.
- **ARC online API key** in `/home/qixinx/ARC-AGI-3-Agents/.env` (`ARC_API_KEY`). Rollout uses the
  ONLINE API at small scale; blocking calls are wrapped in `asyncio.to_thread`.

## Verify in order

**Offline (no GPU, no Miles) — already passing:**
```bash
python3 examples/arc_agi3/env_arc.py            # parser self-test (9/9)
# render/diff need pillow (ARC .venv):
/home/qixinx/ARC-AGI-3-Agents/.venv/bin/python -c "import sys;sys.path.insert(0,'/home/qixinx/miles');\
from examples.arc_agi3.grid_render import render_grid,grid_diff_text;print(render_grid([[0]*64]*64).size)"
# optional live env (uses ARC API): DEMO_GAME=ar25-0c556536 python3 examples/arc_agi3/env_arc.py
```

**M1 gate — SGLang serves Qwen3.5-4B VLM + logprobs** (inside the container, GPU 1): launch
`python3 -m sglang.launch_server --model-path $CKPT --port 30000`, then POST `/generate` with an
`image_data` payload + `"return_logprob": true`; confirm coherent text and
`meta_info.output_token_logprobs`. Hard gate.

**M2 smoke** (inside the container):
```bash
apptainer exec --nv --bind /data,/home/qixinx/miles,/home/qixinx/ARC-AGI-3-Agents \
  /data/user_data/qixinx/images/miles_dev-202606081341.sif \
  bash /home/qixinx/miles/examples/arc_agi3/run_arc_qwen3.5_4B.sh
```
Check: episodes return N turn-samples; each has `loss_mask` all-1 over `response_length`, one
`multimodal_train_inputs` image, `rollout_log_probs` length == `response_length`,
`metadata.{arc_levels,episode_id}`; `arc_advantage` broadcasts one advantage per episode;
rollout→train→update_weights completes; logs show mean reward / win-rate / nonzero-advantage groups.

## Known items to confirm during the first smoke
- Megatron importable in the container (else append its path to the `PYTHONPATH` in the run script).
- Qwen3.5 VLM works with `--megatron-to-hf-mode bridge` (forward+backward on a multimodal batch).
- `--global-batch-size` vs variable turn-counts per episode (tune if a step can't be filled).
- Sparse Δlevels reward likely yields many zero-variance groups on a hard game — expected; this smoke
  validates plumbing, not a learning curve.
