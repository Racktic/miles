"""ARC-AGI-3 interaction environment + response parser for the miles rollout.

Wraps ``arc_rl.arc_env.ArcEnv`` (the validated gym-style ARC env at
``/home/qixinx/ARC-AGI-3-Agents/arc_rl/arc_env.py``) and adds:

- ``parse_response(text, available)`` : pull ``<memory>...</memory>`` and the chosen action
  (``ACTIONk`` / ``RESET``; ``ACTION6`` also carries x,y) out of a model response. This is
  dependency-free (stdlib only) so it can be unit-tested without the ARC SDK or torch.
- ``ArcInteractionEnv`` : ``reset()`` / ``step(name, x, y)`` over ArcEnv with cumulative-levels
  tracking. ``step`` is synchronous (blocking network IO); the rollout wraps it in
  ``asyncio.to_thread`` so it never stalls the event loop.

Response contract the policy is asked to follow (single composed call per turn):

    <memory>...updated world-model memory M_t...</memory>
    <action>ACTION6 x=32 y=45</action>     # or ACTION1..7 / RESET (no args)
"""
from __future__ import annotations

import json
import os
import re
import sys

VALID_ACTIONS = {"RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"}

_MEM_RE = re.compile(r"<memory>(.*?)</memory>", re.DOTALL | re.IGNORECASE)
_ACT_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
_NAME_RE = re.compile(r"\b(RESET|ACTION[1-7])\b", re.IGNORECASE)
_X_RE = re.compile(r"\bx\s*[=:]?\s*(\d{1,2})", re.IGNORECASE)
_Y_RE = re.compile(r"\by\s*[=:]?\s*(\d{1,2})", re.IGNORECASE)
_TWO_INTS_RE = re.compile(r"(\d{1,2})\D+(\d{1,2})")
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _try_json_action(region: str):
    """Best-effort: find a JSON object containing an ``action`` key in ``region``."""
    for m in _JSON_OBJ_RE.finditer(region):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(obj, dict) and "action" in obj:
            name = str(obj.get("action", "")).upper()
            x = obj.get("x")
            y = obj.get("y")
            try:
                x = int(x) if x is not None else None
                y = int(y) if y is not None else None
            except (TypeError, ValueError):
                x = y = None
            return name, x, y
    return None, None, None


def parse_response(text, available=None) -> dict:
    """Parse a model response into ``{memory, action, x, y, valid, reason}``.

    ``available`` is the list of legal action numbers for the current turn (ints, e.g. [1,2,5,6]);
    when given, an action not in it is marked invalid. ``RESET`` is always allowed.
    """
    text = text or ""
    mem_m = _MEM_RE.search(text)
    memory = mem_m.group(1).strip() if mem_m else ""

    act_m = _ACT_RE.search(text)
    region = act_m.group(1) if act_m else text

    # 1) JSON form, then 2) free-text "ACTIONk ... x .. y ..".
    name, x, y = _try_json_action(region)
    if name is None:
        nm = _NAME_RE.search(region) or _NAME_RE.search(text)
        name = nm.group(1).upper() if nm else None

    if name == "ACTION6" and (x is None or y is None):
        # Drop the "ACTION6"/"RESET" token first so its trailing digit isn't read as a coordinate.
        coord_region = _NAME_RE.sub(" ", region)
        xm, ym = _X_RE.search(coord_region), _Y_RE.search(coord_region)
        if xm and ym:
            x, y = int(xm.group(1)), int(ym.group(1))
        else:
            im = _TWO_INTS_RE.search(coord_region)
            if im:
                x, y = int(im.group(1)), int(im.group(2))

    valid, reason = True, ""
    if name is None or name not in VALID_ACTIONS:
        valid, reason = False, "no valid action name found"
    elif name == "ACTION6" and not (x is not None and y is not None and 0 <= x <= 63 and 0 <= y <= 63):
        valid, reason = False, "ACTION6 requires x,y in [0,63]"
    elif available is not None and name != "RESET":
        num = int(name[len("ACTION"):])
        if num not in set(available):
            valid, reason = False, f"{name} not in available actions {sorted(set(available))}"

    return {"memory": memory, "action": name, "x": x, "y": y, "valid": valid, "reason": reason}


def parse_action(text, available=None) -> dict:
    """ACT-phase parser: extract ONLY the chosen action (+ x,y for ACTION6).

    Same action logic as ``parse_response`` but ignores memory (the ACT response carries none).
    Returns ``{action, x, y, valid, reason}``.
    """
    r = parse_response(text, available=available)
    return {k: r[k] for k in ("action", "x", "y", "valid", "reason")}


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)
_MEM_TAG_RE = re.compile(r"</?memory>", re.IGNORECASE)


def clean_memory(text) -> str:
    """Extract the ``<memory>…</memory>`` block (the permanent record M_t) from the REWRITE output;
    the ``<reasoning>`` scratchpad is DISCARDED. Falls back to the whole text (minus wrappers) when
    there is no explicit ``<memory>`` block.
    """
    t = (text or "").strip()
    t = _THINK_RE.sub("", t)
    m = _MEM_RE.search(t)                      # the <memory>…</memory> block IS the record
    if not m:
        return ""                             # no <memory> block → DON'T save raw reasoning; the
        #                                       caller (`clean_memory(...) or memory`) keeps M_{t-1}
    t = m.group(1).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t).strip()
    return t


def _ensure_arc_on_path() -> None:
    """Make ``arc_rl`` importable even when cwd is not the ARC repo (miles runtime)."""
    try:
        import arc_rl.arc_env  # noqa: F401
        return
    except ModuleNotFoundError:
        repo = os.environ.get("ARC_AGI3_REPO", "/home/qixinx/ARC-AGI-3-Agents")
        if repo not in sys.path:
            sys.path.insert(0, repo)


class ArcInteractionEnv:
    """Stateful per-episode ARC env with cumulative-levels (= episode reward) tracking."""

    def __init__(self, game_id: str, max_actions: int = 50, tags=("rl",)):
        _ensure_arc_on_path()
        from arc_rl.arc_env import ArcEnv

        self.game_id = game_id
        self.env = ArcEnv(game_id, max_actions=max_actions, tags=tags)
        self.total_levels: float = 0.0
        self.last_obs = None

    def reset(self) -> dict:
        obs = self.env.reset()
        self.total_levels = float(obs.get("levels", 0) or 0)
        self.last_obs = obs
        return obs

    def step(self, name: str, x=None, y=None):
        """Step the underlying ARC env. Returns ``(obs, reward, done, info)``.

        ``reward`` is the sparse Δlevels for this step; ``total_levels`` accumulates positives
        and equals the episode's final ``levels_completed``.
        """
        obs, reward, done, info = self.env.step(name, x, y)
        self.total_levels += max(float(reward), 0.0)
        self.last_obs = obs
        return obs, reward, done, info

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# Self-test: offline parser checks (no API) + optional live episode (needs ARC_API_KEY).
#   python examples/arc_agi3/env_arc.py            # parser only
#   DEMO_GAME=ar25-0c556536 python examples/arc_agi3/env_arc.py   # + a short live episode
# --------------------------------------------------------------------------------------
def _selftest_parser() -> None:
    cases = [
        ("<memory>walls are value 5</memory><action>ACTION4</action>", [1, 2, 3, 4], "ACTION4", None, None, True),
        ("<memory>m</memory>\n<action>ACTION6 x=32 y=45</action>", [6], "ACTION6", 32, 45, True),
        ("<action>ACTION6 32 45</action>", [6], "ACTION6", 32, 45, True),
        ('reasoning... {"action": "ACTION6", "x": 10, "y": 0}', [6], "ACTION6", 10, 0, True),
        ("I will press <action>ACTION2</action> now", None, "ACTION2", None, None, True),
        ("<memory>x</memory><action>ACTION6 x=70 y=5</action>", [6], "ACTION6", 70, 5, False),  # x>63 -> extracted but invalid
        ("<memory>x</memory><action>ACTION3</action>", [1, 2], "ACTION3", None, None, False),  # not available
        ("no action here at all", None, None, None, None, False),
        ("<action>RESET</action>", [1], "RESET", None, None, True),
    ]
    ok = 0
    for i, (text, avail, e_name, e_x, e_y, e_valid) in enumerate(cases):
        r = parse_response(text, available=avail)
        passed = (r["action"] == e_name and r["x"] == e_x and r["y"] == e_y and r["valid"] == e_valid)
        ok += passed
        flag = "ok " if passed else "FAIL"
        print(f"[{flag}] case {i}: -> action={r['action']} x={r['x']} y={r['y']} valid={r['valid']} reason={r['reason']!r}")
    print(f"parser: {ok}/{len(cases)} passed")
    assert ok == len(cases), "parser self-test failed"


def _selftest_live(game_id: str) -> None:
    env = ArcInteractionEnv(game_id, max_actions=6)
    obs = env.reset()
    print(f"reset: state={obs['state']} levels={obs['levels']} avail={obs['available']} "
          f"grid={len(obs['grid'])}x{len(obs['grid'][0]) if obs['grid'] else 0}")
    from examples.arc_agi3.grid_render import grid_diff_text, render_grid  # local import

    prev = obs["grid"]
    for _ in range(5):
        avail = obs["available"]
        name = f"ACTION{avail[0]}" if avail else "ACTION5"
        obs, r, done, info = env.step(name)
        img = render_grid(obs["grid"])
        diff = grid_diff_text(prev, obs["grid"]).splitlines()[0]
        print(f"step {name}: reward={r} done={done} levels={info['levels']} img={img.size} | {diff}")
        prev = obs["grid"]
        if done:
            break
    print(f"episode total_levels={env.total_levels}")
    env.close()


if __name__ == "__main__":
    _selftest_parser()
    demo = os.environ.get("DEMO_GAME")
    if demo:
        print(f"\n--- live episode on {demo} ---")
        _selftest_live(demo)
    print("OK")
