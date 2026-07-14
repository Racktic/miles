import os
import sys

sys.path.insert(0, "/home/qixinx/ARC-AGI-3-Agents")
sys.path.insert(0, "/home/qixinx/miles")
import arc_rl.arc_env  # noqa: F401

if os.environ.get("ARC_BASE_URL"):
    os.environ["ARC_BASE_URL"] = os.environ["ARC_BASE_URL"].rstrip("/")
from arc_rl.arc_env import ArcEnv
from examples.arc_agi3.object_tracker import ObjectTracker


def cstr(c):
    return "/".join(map(str, c)) if isinstance(c, list) else str(c)


def masks(tagged):
    """The two big L-shaped masks (#3 color=4 right, #5 color=5 left) — the mirror pair."""
    return [(sid, d) for sid, d in tagged if d["size"] > 30 and d["shape"].startswith("mask")]


env = ArcEnv("ar25-0c556536", max_actions=25)
obs = env.reset()
tracker = ObjectTracker()
tagged, bg, _ = tracker.update(obs["grid"])
print(f"=== reset: {len(tagged)} objects ===")
for sid, d in masks(tagged):
    print(f"  #{sid}: color={cstr(d['color'])} size={d['size']} bbox=x[{d['x0']}-{d['x1']}]y[{d['y0']}-{d['y1']}] center=({d['cx']},{d['cy']})")

# Drive ACTION4 (右) repeatedly: #3 moves left (mirror), #5 moves right → they approach the center axis.
prev_grid = obs["grid"]
prev_n = len(tagged)
for i in range(14):
    avail = obs.get("available") or []
    if 4 not in avail:
        print(f"[{i + 1} ACTION4] 不可用 avail={avail}")
        break
    obs, r, done, info = env.step("ACTION4")
    grid = obs["grid"]
    if grid == prev_grid:
        print(f"[{i + 1} ACTION4] 屏幕无变化 (撞墙?)")
        prev_grid = grid
        if done:
            break
        continue
    tagged, bg, changes = tracker.update(grid)
    n = len(tagged)
    ms = masks(tagged)
    gone = [c for c in changes if c[0] == "GONE"]
    appeared = [c for c in changes if c[0] == "APPEARED"]
    reshaped = [c for c in changes if c[0] == "RESHAPED"]
    flag = ""
    if n < prev_n:
        flag = f"   <<< 物体数 {prev_n}->{n} 减少!"
    elif n > prev_n:
        flag = f"   物体数 {prev_n}->{n} 增加"
    mask_info = "  ".join(
        f"#{sid}(c{cstr(d['color'])} x[{d['x0']}-{d['x1']}] sz{d['size']})" for sid, d in ms
    )
    print(f"[{i + 1} ACTION4] objs={n} | masks: {mask_info or '(<2! 可能已合并/遮挡)'}{flag}")
    for c in reshaped:
        print(f"     RESHAPED #{c[1]} {c[2][:14]} -> {c[3][:14]}")
    for c in gone:
        d = c[2]
        print(f"     GONE   #{c[1]} c{cstr(d['color'])} {d['shape'][:12]} sz{d['size']} @({d['cx']},{d['cy']})")
    for c in appeared:
        d = c[2]
        print(f"     APPEAR #{c[1]} c{cstr(d['color'])} {d['shape'][:12]} sz{d['size']} @({d['cx']},{d['cy']})")
    prev_grid = grid
    prev_n = n
    if done:
        print("  (done)")
        break
env.close()
