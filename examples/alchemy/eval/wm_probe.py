"""Ground-truth chemistry -> surface (colour/size/shape) tables for the L_WM logP validation.

Builds, per episode, from the dataset's true chemistry (stone_map / potion_map / bottleneck graph):
  - every latent stone coord -> its surface description + true reward,
  - the full transition table: (stone surface, potion colour) -> resulting stone surface,
  - a programmatic ORACLE-MEMORY (perfect notes) in the same structured style the agent writes.

Surface rendering reuses env_alchemy._stone_desc/_potion_desc so strings match what the agent sees
EXACTLY. Validated: latent->surface via stone_map.apply_inverse reproduces the live env's stones.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/alchemy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # eval/

from env_alchemy import _load_dataset, FIXED_LEVEL, _stone_desc, decode_obs, AlchemyTextEnv  # noqa: E402

_COORDS = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]


def _rw():
    from dm_alchemy.types import stones_and_potions as sp
    return sp.RewardWeights([1, 1, 1], 0, 12)


def _coord_index_map():
    from dm_alchemy.types import stones_and_potions as sp
    return {c: int(sp.LatentStone(np.array(c)).index()) for c in _COORDS}


def stone_surface(sm, rw, c):
    """latent coord tuple -> ('colour size shape', reward:int) exactly as the env renders it.
    desc from the aligned (perceptual) coords; reward is the TRUE latent reward (rw expects the latent
    coords directly — do NOT route it through _stone_desc, whose `value` arg is the normalized obs value)."""
    from dm_alchemy.types import stones_and_potions as sp
    al = sm.apply_inverse(sp.LatentStone(np.array(c)))
    desc, _ = _stone_desc(np.asarray(al.aligned_coords), 0.0)
    rwd = int(round(float(rw(np.array(c)))))
    return desc, rwd


def _potion_colour_map(ep: int, items):
    """(dim,dir) -> colour, harvested by matching the live env's rendered potions to items.potions
    (consistency-checked across all trials)."""
    env = AlchemyTextEnv(episode_index=ep, num_trials=items.num_trials,
                         max_steps_per_trial=20, end_trial_action=True)
    obs = env.reset()
    cmap: dict = {}
    trial = 0
    seen_trials = set()
    # walk trials by ending each one immediately; read potions at each new trial
    while trial < items.num_trials:
        d = decode_obs(obs["vec"])
        t = obs["trial"]
        if t not in seen_trials:
            seen_trials.add(t)
            pots = [p for p in d["potions"] if p["exists"]]
            for slot, p in enumerate(items.potions[t]):
                key = (int(p.dimension), int(p.direction))
                colour = pots[slot]["type"] if slot < len(pots) else None
                if colour is not None:
                    from env_alchemy import _potion_desc
                    col = _potion_desc(colour)
                    if key in cmap and cmap[key] != col:
                        raise ValueError(f"ep{ep} potion colour inconsistent for {key}: {cmap[key]} vs {col}")
                    cmap[key] = col
        # end the trial to advance
        obs, _, done, _ = env.step(0)  # action 0 = end trial
        if done:
            break
        trial = obs["trial"]
    env.close()
    return cmap


def episode_tables(ep: int):
    """Return dict with stone_map, reward fn, adjacency, coord->surface, (dim,dir)->colour, transitions."""
    from dm_alchemy.types import graphs
    chem, items = _load_dataset(FIXED_LEVEL)[ep]
    sm = chem.stone_map
    rw = _rw()
    adj = np.asarray(graphs.convert_graph_to_adj_mat(chem.graph))
    idx = _coord_index_map()
    colour = _potion_colour_map(ep, items)

    def move(c, dim, direction):
        if c[dim] == direction:
            return c
        c2 = c[:dim] + (direction,) + c[dim + 1:]
        return c2 if adj[idx[c]][idx[c2]] else c

    surf = {c: stone_surface(sm, rw, c) for c in _COORDS}
    # full transition table: (c, colour) -> c'
    transitions = []  # (s_desc, s_rwd, colour, d_desc, d_rwd, changed:bool, blocked:bool)
    for c in _COORDS:
        for (dim, direction), col in colour.items():
            c2 = move(c, dim, direction)
            would = (c[dim] != direction)            # potion would act (stone not already at target)
            blocked = would and (c2 == c)            # would act but graph edge missing
            transitions.append((surf[c][0], surf[c][1], col, surf[c2][0], surf[c2][1],
                                c2 != c, blocked))
    return {"surf": surf, "colour": colour, "move": move, "transitions": transitions,
            "adj": adj, "idx": idx, "num_trials": items.num_trials}


def _attr_changed(a: str, b: str):
    """Which of (colour,size,shape) differs between two 'colour size shape' descs, and from->to."""
    aw, bw = a.split(), b.split()
    names = ["colour", "size", "shape"]
    for i in range(3):
        if aw[i] != bw[i]:
            return names[i], aw[i], bw[i]
    return None, None, None


def oracle_memory(ep: int) -> str:
    """Programmatic PERFECT memory from ground truth, in the structured style the agent writes."""
    T = episode_tables(ep)
    surf = T["surf"]
    # --- Potion Effects: one rule per colour, derived from the transition table ---
    by_colour: dict = {}
    for s_desc, s_rwd, col, d_desc, d_rwd, changed, blocked in T["transitions"]:
        by_colour.setdefault(col, []).append((s_desc, d_desc, changed, blocked))
    lines = ["### Potion Effects"]
    for col in sorted(by_colour):
        rows = by_colour[col]
        # the attribute this colour changes (from any actually-changing transition)
        attr = frm = to = None
        for s_desc, d_desc, changed, _ in rows:
            if changed:
                attr, frm, to = _attr_changed(s_desc, d_desc)
                break
        if attr is None:
            lines.append(f"- **{col}**: no effect on any stone (inert this episode).")
            continue
        blocked_srcs = [s for s, d, ch, bl in rows if bl]
        rule = f"- **{col}**: changes a stone's {attr} from {frm} to {to} (acts only when the stone is {frm})."
        if blocked_srcs:
            rule += f" Blocked (no effect) for: {', '.join(sorted(set(blocked_srcs)))}."
        lines.append(rule)
    # --- Highest Reward Combination + full reward landscape ---
    best_c = max(_COORDS, key=lambda c: surf[c][1])
    lines.append("")
    lines.append("### Highest Reward Combination")
    lines.append(f"Highest reward is {surf[best_c][1]:+d}, achieved by **{surf[best_c][0]}**.")
    lines.append("Reward of every stone type:")
    for c in sorted(_COORDS, key=lambda c: -surf[c][1]):
        lines.append(f"- {surf[c][0]}: {surf[c][1]:+d}")
    return "\n".join(lines)


def potion_effects(ep: int) -> dict:
    """{colour: (attr, frm, to)} ground-truth surface effect per potion, or (None,None,None) if inert.
    Used to objectively score how many rules a real memory got right."""
    T = episode_tables(ep)
    by_colour: dict = {}
    for s_desc, s_rwd, col, d_desc, d_rwd, changed, blocked in T["transitions"]:
        by_colour.setdefault(col, [])
        if changed:
            by_colour[col].append((s_desc, d_desc))
    out = {}
    for col, rows in by_colour.items():
        if not rows:
            out[col] = (None, None, None)
        else:
            out[col] = _attr_changed(rows[0][0], rows[0][1])
    return out


def oracle_memory_table(ep: int) -> str:
    """Maximally explicit PERFECT memory: the full transition table listed directly, so the reader only
    has to LOOK UP the matching (stone, potion) line and copy the result — zero inference. Strongest
    positive control: if even this doesn't raise logP, it's the scorer/reader, not the oracle phrasing."""
    T = episode_tables(ep)
    lines = ["### Transition table (stone + potion -> resulting stone)"]
    for s_desc, s_rwd, col, d_desc, d_rwd, changed, blocked in T["transitions"]:
        s = f"{s_desc} with reward {s_rwd:+d}"
        d = f"{d_desc} with reward {d_rwd:+d}"
        lines.append(f"- {s} + {col} -> {d}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=55)
    ap.add_argument("--table", action="store_true", help="print the table-style oracle instead")
    args = ap.parse_args()
    print(f"========== ORACLE-MEMORY ({'table' if args.table else 'rules'}) for episode {args.ep} ==========\n")
    print(oracle_memory_table(args.ep) if args.table else oracle_memory(args.ep))
    print("\n========== a few transitions (sanity) ==========")
    T = episode_tables(args.ep)
    for row in T["transitions"][:12]:
        s_desc, s_rwd, col, d_desc, d_rwd, changed, blocked = row
        tag = "BLOCKED" if blocked else ("changed" if changed else "no-op")
        print(f"  {s_desc} ({s_rwd:+d}) + {col} -> {d_desc} ({d_rwd:+d})  [{tag}]")
