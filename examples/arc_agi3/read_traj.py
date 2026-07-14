"""Pretty-print ARC-AGI-3 rollout trajectories dumped by arc_rollout into ARC_TRAJ_DIR.

Each episode is one ``ep_<index>.json`` (memory / action / reward / diff per turn; no images).

Usage:
  python3 examples/arc_agi3/read_traj.py [TRAJ_DIR] [--full] [--limit N]
    TRAJ_DIR  defaults to $ARC_TRAJ_DIR or /data/user_data/qixinx/arc_traj
    --full    print full memory text (default truncates to 300 chars)
    --limit N show only the N most recent episodes
"""
import glob
import json
import os
import sys


def main():
    argv = sys.argv[1:]
    full = "--full" in argv
    limit = None
    skip = -1
    if "--limit" in argv:
        i = argv.index("--limit")
        skip = i + 1  # the value after --limit is the count, NOT a positional traj_dir
        try:
            limit = int(argv[skip])
        except (ValueError, IndexError):
            limit = None
    pos = [a for j, a in enumerate(argv) if not a.startswith("-") and j != skip]
    if pos:
        traj_dir = pos[0]
    elif os.environ.get("ARC_TRAJ_DIR"):
        traj_dir = os.environ["ARC_TRAJ_DIR"]
    else:  # default = MOST RECENT run's traj dir under logs/<run-id>/traj
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        runs = sorted(glob.glob(os.path.join(base, "*", "traj")), key=os.path.getmtime)
        traj_dir = runs[-1] if runs else os.path.join(base, "traj")

    files = sorted(glob.glob(os.path.join(traj_dir, "ep_*.json")), key=os.path.getmtime)
    if not files:
        print(f"(no ep_*.json found in {traj_dir})")
        return
    if limit:
        files = files[-limit:]

    won = 0
    for fp in files:
        try:
            ep = json.load(open(fp))
        except Exception as e:
            print(f"  skip {fp}: {e}")
            continue
        won += ep.get("final_levels", 0) > 0
        print("=" * 100)
        print(f"episode {ep.get('episode_id')}  game={ep.get('game_id')}  "
              f"final_levels={ep.get('final_levels')}  turns={ep.get('num_turns')}")
        for t in ep.get("turns", []):
            print("-" * 100)
            if t.get("raw_act") is not None or t.get("raw_rewrite") is not None:
                # two-phase trajectory: ACT (decide) -> REWRITE (memory M_t)
                print(f"  turn {t.get('turn')}  action={t.get('action')}  valid={t.get('valid')}  "
                      f"reward={t.get('reward')}  levels_after={t.get('levels_after')}  "
                      f"act_finish={t.get('act_finish')} rw_finish={t.get('rw_finish')}")
                if t.get("note"):
                    print(f"    note: {t['note']}")
                act = (t.get("raw_act") or "").strip()
                mem = (t.get("memory") or "").strip()
                act = act if full else (act[:400] + ("…" if len(act) > 400 else ""))
                mem = mem if full else (mem[:400] + ("…" if len(mem) > 400 else ""))
                print(f"    ACT: {act}")
                print(f"    MEMORY(M_t): {mem}")
            else:
                # legacy single-phase trajectory
                raw = (t.get("raw_response") or t.get("memory") or "").strip()
                label = "raw_response" if t.get("raw_response") else "memory(parsed only)"
                shown = raw if full else (raw[:400] + ("…" if len(raw) > 400 else ""))
                print(f"  turn {t.get('turn')}  action={t.get('action')}  valid={t.get('valid')}  "
                      f"reward={t.get('reward')}  levels_after={t.get('levels_after')}  finish={t.get('finish')}")
                if t.get("note"):
                    print(f"    note: {t['note']}")
                print(f"    {label}: {shown}")
    print("=" * 100)
    print(f"{len(files)} episode(s) in {traj_dir};  {won} reached a level (final_levels>0)")


if __name__ == "__main__":
    main()
