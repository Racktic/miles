"""Diagnostic: time WHERE the ideal-observer oracle spends time — precomputed-map load vs the
belief-state search — and whether minimise_world_states=True makes it tractable. Stage-by-stage,
flushed, so a `timeout` wrapper tells us exactly which stage is the bottleneck.
"""
import os
import sys
import time
import types

sys.path.insert(0, "examples/alchemy")
sys.path.insert(0, "examples/alchemy/eval")

import numpy as _np
if not getattr(_np.stack, "_accepts_gen", False):
    _o = _np.stack
    def _s(a, *x, **k):
        if isinstance(a, types.GeneratorType):
            a = list(a)
        return _o(a, *x, **k)
    _s._accepts_gen = True
    _np.stack = _s

LEVEL = "perceptual_mapping_randomized_with_random_bottleneck"
MINIMISE = os.environ.get("DIAG_MINIMISE", "1") == "1"

t = time.time()
from dm_alchemy.ideal_observer import precomputed_maps  # noqa: E402
print(f"[diag] import precomputed_maps: {time.time()-t:.1f}s", flush=True)

t = time.time()
pc = precomputed_maps.load_from_level_name(LEVEL)
print(f"[diag] LOAD load_from_level_name: {time.time()-t:.1f}s", flush=True)

from env_alchemy import _load_dataset  # noqa: E402
from dm_alchemy import symbolic_alchemy_bots  # noqa: E402
from dm_alchemy import symbolic_alchemy_trackers as T  # noqa: E402
from dm_alchemy.types import stones_and_potions  # noqa: E402

chem, items = _load_dataset(LEVEL)[0]
rw = stones_and_potions.RewardWeights([1, 1, 1], 0, 12)

def addt(env):
    env.add_trackers({
        T.AddMatrixEventTracker.NAME: T.AddMatrixEventTracker(),
        T.ScoreTracker.NAME: T.ScoreTracker(rw),
        T.BeliefStateTracker.NAME: T.BeliefStateTracker(pc, env, None)})

t = time.time()
res = symbolic_alchemy_bots.get_multi_trial_ideal_observer_reward(
    items, chem, rw, pc, MINIMISE, addt)
print(f"[diag] IDEAL OBSERVER (minimise_world_states={MINIMISE}) 1 episode: {time.time()-t:.1f}s",
      flush=True)
print(f"[diag] per_trial = {res['score']['per_trial']}", flush=True)
print("[diag] DONE", flush=True)
