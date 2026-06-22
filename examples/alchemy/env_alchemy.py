"""Symbolic DeepMind Alchemy interaction environment + text representation + response parser.

Wraps ``dm_alchemy.symbolic_alchemy`` (a meta-RL benchmark: a latent causal "chemistry" —
which potion moves a stone along which perceptual axis, plus a perceptual rotation — is resampled
per *episode*; each episode has ``num_trials`` trials over fresh stones/potions of the SAME hidden
chemistry, so the agent must infer the chemistry online and exploit it). We expose it to the miles
two-phase (ACT -> REWRITE) text rollout, mirroring ``examples/arc_agi3/env_arc.py``:

- ``obs_to_text`` / ``obs_diff_text`` : render the 39-d symbolic obs vector (and the before/after
  transition) as compact TEXT — no perception, the transition ``d_t`` is exact and noise-free.
- ``parse_action(text, obs)`` : pull the chosen move ("put stoneI into potionJ / cauldron", "no-op")
  out of a model response and map it to the env's integer action. Stdlib+numpy only (no torch / no
  miles), so it unit-tests offline.
- ``clean_memory`` : same ``<memory>…</memory>`` contract as the ARC path (reused verbatim shape).
- ``AlchemyTextEnv`` : ``reset()`` / ``step(action_int)`` over the symbolic env with cumulative
  reward tracking and trial/step bookkeeping.

obs layout (observe_used=True, num_axes=3, MAX_STONES=3, MAX_POTIONS=12  -> 39 dims):
    stones : 3 x [axis0, axis1, axis2, value, used]      (15)
    potions: 12 x [type, used]                            (24)
  ``used==0`` means the slot is still present; the default (``used==1``, coords==2) means empty/used.

action mapping (end_trial_action=False -> exactly one special action, no-op=0):
    a==0                      -> no-op
    a in [1,39]: altered=a-1; stone=altered//13; rem=altered%13
                 rem==0 -> stone into cauldron ; rem in [1,12] -> stone into potion (rem-1)
  inverse: int = stone*13 + (potion+1) + 1     (potion=-1 == cauldron)
"""
from __future__ import annotations

import os
import re

import numpy as np

# ---- fixed structural constants (match dm_alchemy symbolic_alchemy defaults) -------------------
MAX_STONES = 3
MAX_POTIONS = 12
NUM_AXES = 3
STONE_FEATS = NUM_AXES + 2          # axis0,axis1,axis2, value, used = 5
POTION_FEATS = 2                    # type, used
POTIONS_AND_CAULDRON = MAX_POTIONS + 1   # 13
NUM_SPECIAL = 1                     # no-op only (end_trial_action=False)
STONE_BLOCK = MAX_STONES * STONE_FEATS   # 15
OBS_DIM = STONE_BLOCK + MAX_POTIONS * POTION_FEATS  # 39

DEFAULT_LEVEL = "alchemy/perceptual_mapping_randomized_with_rotation_and_random_bottleneck"
# Official prebuilt dataset packaged inside dm_alchemy: 1000 fixed (chemistry, episode_items) pairs,
# each 10 trials x 3 stones x 12 potions. This is the *no-rotation* level (one fewer perceptual
# confound than DEFAULT_LEVEL) — friendlier for bootstrapping co-learning, and reproducible.
FIXED_LEVEL = "perceptual_mapping_randomized_with_random_bottleneck"

VALID_KINDS = {"potion", "cauldron", "noop"}

_DATASET_CACHE: dict = {}


def _load_dataset(level_name: str = FIXED_LEVEL):
    """Load & cache the official prebuilt (chemistry, episode_items) list shipped with dm_alchemy.

    Returns ``List[Tuple[Chemistry, EpisodeItems]]`` (length 1000). Cached per level so constructing
    many envs doesn't re-parse the 250KB proto each time."""
    if level_name not in _DATASET_CACHE:
        from dm_alchemy.encode import chemistries_proto_conversion as cpc
        # Prefer the in-repo copy (examples/alchemy/data/<level>/chemistries) so the dataset is
        # version-controlled and reproducible regardless of where dm_alchemy is installed; fall back
        # to the copy shipped inside the dm_alchemy package. Both are byte-identical.
        repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "data", level_name, "chemistries")
        if os.path.exists(repo_path):
            path = repo_path
        else:
            import dm_alchemy
            path = os.path.join(os.path.dirname(dm_alchemy.__file__),
                                "chemistries", level_name, "chemistries")
        _DATASET_CACHE[level_name] = cpc.load_chemistries_and_items(path)
    return _DATASET_CACHE[level_name]


def dataset_size(level_name: str = FIXED_LEVEL) -> int:
    """Number of prebuilt episodes available (for splitting train/eval over episode indices)."""
    return len(_load_dataset(level_name))


# ==============================================================================================
# obs decode + text rendering  (pure functions on the 39-d vector; no env / torch dependency)
# ==============================================================================================
def decode_obs(vec) -> dict:
    """39-d float vector -> {'stones': [...], 'potions': [...]} with existence flags decoded."""
    v = np.asarray(vec, dtype=float).reshape(-1)
    assert v.shape[0] == OBS_DIM, f"expected {OBS_DIM}-d obs, got {v.shape}"
    stones = []
    for i in range(MAX_STONES):
        b = i * STONE_FEATS
        used = v[b + NUM_AXES + 1]
        stones.append({
            "idx": i,
            "exists": used == 0.0,
            "axes": v[b:b + NUM_AXES].copy(),
            "value": float(v[b + NUM_AXES]),
        })
    potions = []
    for j in range(MAX_POTIONS):
        b = STONE_BLOCK + j * POTION_FEATS
        used = v[b + 1]
        potions.append({"idx": j, "exists": used == 0.0, "type": float(v[b])})
    return {"stones": stones, "potions": potions}


# ---- natural-language rendering (Sawyer et al. 2025 style) ----
# axis value -> perceptual word; stone value -> true cauldron reward; potion type -> colour.
# no-rotation levels use axes in {-1,+1}; the 0 ("medium/grey/flat") rows cover rotation levels.
_SIZE  = {-1: "small", 0: "medium", 1: "large"}    # axis 0
_HUE   = {-1: "blue",  0: "grey",   1: "purple"}   # axis 1  (stone colour)
_SHAPE = {-1: "round", 0: "flat",   1: "pointy"}   # axis 2
_VALUE_TO_REWARD = {-1.0: -3, -0.33: -1, 0.33: 1, 1.0: 15}   # value's 4 tiers -> integer reward
_POTION_COLORS = {-1.0: "green", -0.67: "red", -0.33: "yellow",
                  0.0: "orange", 0.33: "pink", 0.67: "turquoise"}   # potion type (colour id) -> name


def _stone_desc(axes, value):
    """(axes, value) -> ('blue small pointy', reward_int): paper-style appearance + integer reward."""
    a = [int(round(x)) for x in axes]
    desc = f"{_HUE.get(a[1], '?')} {_SIZE.get(a[0], '?')} {_SHAPE.get(a[2], '?')}"
    rwd = _VALUE_TO_REWARD.get(round(value, 2))
    if rwd is None:                          # fallback for rotation / other tiers
        rwd = int(round(value * 15))
    return desc, rwd


def _potion_desc(ptype) -> str:
    return _POTION_COLORS.get(round(ptype, 2), f"colour{int(round((ptype + 1) * 3))}")


def obs_to_text(vec, trial=None, step=None, num_trials=None, max_steps=None) -> str:
    """Render the current state in the natural-language style of Sawyer et al. (2025):
    each stone as '<colour> <size> <shape> with reward <±N>', each potion as its colour."""
    d = decode_obs(vec)
    lines = []
    if trial is not None:
        head = f"Trial {trial + 1}/{num_trials}" if num_trials else f"Trial {trial + 1}"
        if step is not None:
            head += f", step {step + 1}" + (f"/{max_steps}" if max_steps else "")
        lines.append(head)
    lines.append("You see the following stones:")
    any_stone = False
    for s in d["stones"]:
        if s["exists"]:
            any_stone = True
            desc, rwd = _stone_desc(s["axes"], s["value"])
            lines.append(f"  {s['idx']}: {desc} with reward {rwd:+d}")
    if not any_stone:
        lines.append("  (none left this trial)")
    lines.append("You see the following potions:")
    avail = [p for p in d["potions"] if p["exists"]]
    if avail:
        for p in avail:
            lines.append(f"  {p['idx']}: {_potion_desc(p['type'])}")
    else:
        lines.append("  (none left this trial)")
    return "\n".join(lines)


def obs_diff_text(before, after, reward, new_trial=False) -> str:
    """Render the transition d_t (before -> after) + reward in the same natural-language style."""
    r = f"Reward gained this step: {int(round(float(reward))):+d}"
    if new_trial:
        return (r + "\nA new trial began: the stones and potions were refreshed (same hidden "
                "chemistry, new items). Do not compare items across this boundary.")
    db, da = decode_obs(before), decode_obs(after)
    lines = []
    for sb, sa in zip(db["stones"], da["stones"]):
        if sb["exists"] and not sa["exists"]:
            lines.append(f"Stone {sb['idx']} was removed (placed into a potion or the cauldron).")
        elif sb["exists"] and sa["exists"]:
            moved = not np.allclose(sb["axes"], sa["axes"]) or abs(sb["value"] - sa["value"]) > 1e-6
            if moved:
                d_before, _ = _stone_desc(sb["axes"], sb["value"])
                d_after, rwd = _stone_desc(sa["axes"], sa["value"])
                lines.append(f"Stone {sb['idx']} changed: {d_before} -> {d_after} "
                             f"(its reward is now {rwd:+d}).")
    for pb, pa in zip(db["potions"], da["potions"]):
        if pb["exists"] and not pa["exists"]:
            lines.append(f"Potion {pb['idx']} ({_potion_desc(pb['type'])}) was used up.")
    if not lines:
        lines.append("Nothing changed (it was a no-op, or the move was rejected).")
    return r + "\n" + "\n".join(lines)


# ==============================================================================================
# action: integer <-> semantic, and response parsing
# ==============================================================================================
def semantic_to_int(stone_ind: int, potion_ind: int, num_special: int = NUM_SPECIAL) -> int:
    """(stone, potion) -> env integer action. ``potion_ind == -1`` means the cauldron.
    ``num_special`` = 1 (no-op only) or 2 (end-trial + no-op, when ``end_trial_action`` is on)."""
    return stone_ind * POTIONS_AND_CAULDRON + (potion_ind + 1) + num_special


def int_to_semantic(action_int: int, num_special: int = NUM_SPECIAL) -> dict:
    """Inverse of ``semantic_to_int`` (for logging / verification).
    With ``num_special == 2``: 0 = end-trial, 1 = no-op; with 1: 0 = no-op."""
    if action_int < num_special:
        if num_special == 2 and action_int == 0:
            return {"kind": "end_trial"}
        return {"kind": "noop"}
    altered = action_int - num_special
    stone = altered // POTIONS_AND_CAULDRON
    rem = altered % POTIONS_AND_CAULDRON
    if rem == 0:
        return {"kind": "cauldron", "stone": stone}
    return {"kind": "potion", "stone": stone, "potion": rem - 1}


_ACT_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_MEM_RE = re.compile(r"<memory>(.*?)</memory>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_STONE_RE = re.compile(r"stone\s*[#:<]?\s*([0-2])", re.IGNORECASE)   # tolerate 'stone <0>' (placeholder copy)
_POTION_RE = re.compile(r"potion\s*[#:<]?\s*(\d{1,2})", re.IGNORECASE)
_CAULDRON_RE = re.compile(r"cauldron", re.IGNORECASE)
_NOOP_RE = re.compile(r"\bno[\s_-]?op\b|do nothing|\bskip\b|\bpass\b|\bwait\b", re.IGNORECASE)
_ENDTRIAL_RE = re.compile(r"end\s+(the\s+)?trial|end[\s_-]?trial", re.IGNORECASE)


def parse_action(text, obs=None, end_trial: bool = False) -> dict:
    """Parse a model response into a concrete move.

    Returns ``{action, kind, stone, potion, valid, reason, desc}`` where ``action`` is the integer
    to pass to ``AlchemyTextEnv.step``. ``obs`` rejects moves referencing an absent stone/potion.
    ``end_trial`` enables the end-trial action (num_special=2: int 0 = end-trial, 1 = no-op);
    otherwise num_special=1 (int 0 = no-op).
    """
    text = text or ""
    ms = list(_ACT_RE.finditer(text))   # model reasons first, then emits the tag -> take the LAST one
    region = ms[-1].group(1) if ms else text
    num_special = 2 if end_trial else 1

    if end_trial and _ENDTRIAL_RE.search(region) and not _STONE_RE.search(region):
        return {"action": 0, "kind": "end_trial", "stone": None, "potion": None,
                "valid": True, "reason": "", "desc": "end the trial"}
    if _NOOP_RE.search(region) and not _STONE_RE.search(region):
        return {"action": num_special - 1, "kind": "noop", "stone": None, "potion": None,
                "valid": True, "reason": "", "desc": "no-op"}

    sm = _STONE_RE.search(region)
    if sm is None:
        return {"action": None, "kind": None, "stone": None, "potion": None,
                "valid": False, "reason": "no stone referenced", "desc": ""}
    stone = int(sm.group(1))

    to_cauldron = bool(_CAULDRON_RE.search(region))
    pm = _POTION_RE.search(region)
    if to_cauldron and pm is None:
        kind, potion = "cauldron", -1
    elif pm is not None:
        kind, potion = "potion", int(pm.group(1))
    else:
        return {"action": None, "kind": None, "stone": stone, "potion": None,
                "valid": False, "reason": "no potion index or 'cauldron' given", "desc": ""}

    # range + existence validation against the live obs
    reason = ""
    if kind == "potion" and not (0 <= potion < MAX_POTIONS):
        reason = f"potion index {potion} out of range [0,{MAX_POTIONS - 1}]"
    elif obs is not None:
        d = decode_obs(obs["vec"] if isinstance(obs, dict) else obs)
        if not d["stones"][stone]["exists"]:
            reason = f"stone{stone} is not present"
        elif kind == "potion" and not d["potions"][potion]["exists"]:
            reason = f"potion{potion} is not present"

    desc = (f"put stone{stone} into cauldron" if kind == "cauldron"
            else f"put stone{stone} into potion{potion}")
    if reason:
        return {"action": None, "kind": kind, "stone": stone,
                "potion": (None if kind == "cauldron" else potion),
                "valid": False, "reason": reason, "desc": desc}
    return {"action": semantic_to_int(stone, potion, num_special), "kind": kind, "stone": stone,
            "potion": (None if kind == "cauldron" else potion),
            "valid": True, "reason": "", "desc": desc}


def clean_memory(text) -> str:
    """Extract the ``<memory>…</memory>`` block (M_t) from a REWRITE output; discard the rest.
    No ``<memory>`` block -> return "" so the caller keeps M_{t-1} (same contract as env_arc)."""
    t = (text or "").strip()
    t = _THINK_RE.sub("", t)
    m = _MEM_RE.search(t)
    if not m:
        return ""
    t = m.group(1).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t).strip()
    return t


# ==============================================================================================
# environment wrapper
# ==============================================================================================
class AlchemyTextEnv:
    """Stateful per-episode symbolic Alchemy env with cumulative reward + trial/step bookkeeping."""

    def __init__(self, episode_index=None, level_name: str | None = None, num_trials: int = 10,
                 max_steps_per_trial: int = 20, seed: int = 0, end_trial_action: bool = False):
        """Two construction modes:

        - ``episode_index`` given (DEFAULT use): load that fixed (chemistry, items) pair from the
          official prebuilt dataset (``FIXED_LEVEL``, 1000 episodes) — reproducible, paper-aligned.
        - ``episode_index=None``: procedurally generate a random chemistry from ``seed`` on
          ``level_name`` (default ``DEFAULT_LEVEL``, the with-rotation level).

        ``end_trial_action`` (eval-only, TMLR-aligned): lets the agent end a trial early; this shifts
        the integer action space (int 0 = end-trial, 1 = no-op, 2+ = stone/potion). Training leaves it
        False so the action mapping (int 0 = no-op, 1+ = stone/potion) is unchanged.
        """
        from dm_alchemy import symbolic_alchemy
        if episode_index is not None:
            chem, items = _load_dataset(level_name or FIXED_LEVEL)[int(episode_index)]
            self._env = symbolic_alchemy.get_symbolic_alchemy_fixed(
                episode_items=items, chemistry=chem, max_steps_per_trial=max_steps_per_trial,
                end_trial_action=end_trial_action)
            self.num_trials = items.num_trials
            self.episode_index = int(episode_index)
        else:
            self._env = symbolic_alchemy.get_symbolic_alchemy_level(
                level_name=level_name or DEFAULT_LEVEL, num_trials=num_trials,
                max_steps_per_trial=max_steps_per_trial, seed=seed,
                end_trial_action=end_trial_action)
            self.num_trials = num_trials
            self.episode_index = None
        self.end_trial_action = end_trial_action
        self.num_special = 2 if end_trial_action else 1
        self.max_steps = max_steps_per_trial
        self.total_reward = 0.0
        self.per_trial_scores: list = []           # cumulative cauldron reward per trial
        self.last_vec = None

    def _obs(self, new_trial: bool) -> dict:
        trial = int(getattr(self._env, "trial_number", 0))
        step = int(getattr(self._env, "_steps_this_trial", 0))
        score = self.per_trial_scores[-1] if self.per_trial_scores else 0.0
        return {"vec": self.last_vec, "trial": trial, "step": step, "trial_score": score,
                "num_trials": self.num_trials, "max_steps": self.max_steps, "new_trial": new_trial}

    def reset(self) -> dict:
        ts = self._env.reset()
        self.last_vec = np.asarray(ts.observation["symbolic_obs"], dtype=np.float32)
        self.total_reward = 0.0
        self.per_trial_scores = [0.0]
        return self._obs(new_trial=True)

    def step(self, action_int):
        """Step the symbolic env. Returns ``(obs, reward, done, info)``.

        ``reward`` is this step's reward (cauldron drops give the stone's value; 0 otherwise).
        A trial boundary is surfaced via ``obs['new_trial']`` so the REWRITE prompt can skip the
        (meaningless) cross-boundary diff."""
        trial_before = int(getattr(self._env, "trial_number", 0))
        ts = self._env.step(int(action_int))
        self.last_vec = np.asarray(ts.observation["symbolic_obs"], dtype=np.float32)
        reward = float(ts.reward or 0.0)
        self.total_reward += reward
        done = bool(ts.last())
        new_trial = (int(getattr(self._env, "trial_number", 0)) != trial_before) and not done
        # This step's reward belongs to the trial that just ran (cauldron drops resolve before the
        # boundary), so accumulate into the current bucket, then open a fresh bucket on a new trial.
        if self.per_trial_scores:
            self.per_trial_scores[-1] += reward
        if new_trial:
            self.per_trial_scores.append(0.0)
        info = {"step_type": int(ts.step_type), "total_reward": self.total_reward}
        return self._obs(new_trial=new_trial), reward, done, info

    def close(self) -> None:
        try:
            if hasattr(self._env, "close"):
                self._env.close()
        except Exception:
            pass


# ==============================================================================================
# offline self-test:  python examples/alchemy/env_alchemy.py
#   - parser round-trip (no env)         always
#   - live episode (obs->text, step, diff)  unless ALCHEMY_NO_ENV=1
# ==============================================================================================
def _selftest_parser() -> None:
    cases = [
        ("<action>put stone0 into cauldron</action>", "cauldron", 0, None, True),
        ("<action>put stone2 into potion5</action>", "potion", 2, 5, True),
        ("I'll drop stone1 in potion 11.", "potion", 1, 11, True),
        ("<action>stone0 -> cauldron</action>", "cauldron", 0, None, True),
        ("<action>no-op</action>", "noop", None, None, True),
        ("<action>put stone1 into potion12</action>", "potion", 1, 12, False),  # potion OOB
        ("<action>just think harder</action>", None, None, None, False),        # no stone
    ]
    # int round-trip against the documented mapping
    assert semantic_to_int(0, -1) == 1 and semantic_to_int(0, 0) == 2
    assert semantic_to_int(1, -1) == 14 and semantic_to_int(2, -1) == 27
    for s in range(MAX_STONES):
        for p in range(-1, MAX_POTIONS):
            sem = int_to_semantic(semantic_to_int(s, p))
            assert sem["stone"] == s and sem.get("potion", -1) == p, (s, p, sem)
    ok = 0
    for i, (text, e_kind, e_stone, e_potion, e_valid) in enumerate(cases):
        r = parse_action(text, obs=None)
        passed = (r["kind"] == e_kind and r["stone"] == e_stone
                  and r["potion"] == e_potion and r["valid"] == e_valid)
        ok += passed
        print(f"[{'ok ' if passed else 'FAIL'}] case {i}: kind={r['kind']} stone={r['stone']} "
              f"potion={r['potion']} valid={r['valid']} action={r['action']} reason={r['reason']!r}")
    print(f"parser+mapping: {ok}/{len(cases)} cases passed")
    assert ok == len(cases), "parser self-test failed"


def _selftest_live() -> None:
    print(f"\nprebuilt dataset size: {dataset_size()} episodes")
    env = AlchemyTextEnv(episode_index=0, max_steps_per_trial=6)  # official prebuilt episode 0
    obs = env.reset()
    print("--- reset ---")
    print(obs_to_text(obs["vec"], obs["trial"], obs["step"], obs["num_trials"], obs["max_steps"]))

    # craft a few model-style responses, parse -> step -> render the exact transition
    responses = [
        "<action>put stone0 into potion0</action>",   # transform stone0
        "<action>put stone0 into potion1</action>",   # transform it again
        "<action>put stone0 into cauldron</action>",  # cash it in
        "<action>put stone1 into cauldron</action>",
        "<action>no-op</action>",
    ]
    for resp in responses:
        before = obs["vec"]
        parsed = parse_action(resp, obs=obs)
        print(f"\n>>> {resp!r}\n    parsed: {parsed['desc']} -> action_int={parsed['action']} "
              f"valid={parsed['valid']} ({parsed['reason']})")
        if not parsed["valid"]:
            print("    (rejected; screen unchanged)")
            continue
        obs, reward, done, info = env.step(parsed["action"])
        print("    " + obs_diff_text(before, obs["vec"], reward, new_trial=obs["new_trial"])
              .replace("\n", "\n    "))
        if done:
            print("    [episode done]")
            break
    print(f"\nepisode total_reward={env.total_reward}")
    env.close()


if __name__ == "__main__":
    _selftest_parser()
    if os.environ.get("ALCHEMY_NO_ENV") != "1":
        print("\n=== live episode ===")
        _selftest_live()
    print("\nOK")
