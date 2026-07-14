import os
import sys
from collections import Counter

sys.path.insert(0, "/home/qixinx/ARC-AGI-3-Agents")
sys.path.insert(0, "/home/qixinx/miles")
import arc_rl.arc_env  # noqa: F401

if os.environ.get("ARC_BASE_URL"):
    os.environ["ARC_BASE_URL"] = os.environ["ARC_BASE_URL"].rstrip("/")
from arc_rl.arc_env import ArcEnv
from examples.arc_agi3.object_tracker import ObjectTracker


def cstr(c):
    return "/".join(map(str, c)) if isinstance(c, list) else str(c)


env = ArcEnv("ar25-0c556536", max_actions=40)
obs = env.reset()
tracker = ObjectTracker()
tagged, bg, _ = tracker.update(obs["grid"])
print(f"=== reset: {len(tagged)} objects (bg={bg}) ===")

# 覆盖四个方向，并一路上推/左推制造"可动物体撞到不动物体"的重合
seq = (["ACTION1"] * 3 + ["ACTION2"] * 2          # 上3 下2
       + ["ACTION3"] * 2 + ["ACTION4"] * 3         # 左2 右3
       + ["ACTION1"] * 8                           # 一路上推（撞顶 / 可能与上方物体重合）
       + ["ACTION3"] * 8)                          # 一路左推

prev_grid = obs["grid"]
prev_n = len(tagged)
ACT_NAME = {1: "上 dy-3", 2: "下 dy+3", 3: "左 dx-3", 4: "右 dx+3", 5: "?", 7: "?"}

for i, act in enumerate(seq):
    avail = obs.get("available") or []
    a = int(act[-1])
    if a not in avail:
        print(f"[{i + 1:2d} {act} {ACT_NAME.get(a,'')}] 不在 available={avail}, skip")
        continue
    obs, r, done, info = env.step(act)
    grid = obs["grid"]
    if grid == prev_grid:
        print(f"[{i + 1:2d} {act} {ACT_NAME.get(a,'')}] 屏幕无变化")
        prev_grid = grid
        if done:
            break
        continue
    tagged, bg, changes = tracker.update(grid)
    n = len(tagged)
    moved = [c for c in changes if c[0] == "MOVED"]
    reshaped = [c for c in changes if c[0] == "RESHAPED"]
    gone = [c for c in changes if c[0] == "GONE"]
    appeared = [c for c in changes if c[0] == "APPEARED"]
    deltas = Counter((c[2], c[3]) for c in moved)
    main = deltas.most_common(1)[0][0] if deltas else None

    flag = ""
    if n < prev_n:
        flag = f"  <<< 物体数 {prev_n}->{n} 减少 (合并 or 移出?)"
    elif n > prev_n:
        flag = f"  物体数 {prev_n}->{n} 增加"
    print(f"[{i + 1:2d} {act} {ACT_NAME.get(a,'')}] objs={n} 主位移={main} "
          f"MOVED={len(moved)} RESHAPE={len(reshaped)} GONE={len(gone)} APPEAR={len(appeared)}{flag}")
    if len(deltas) > 1:
        print(f"      ⚠ 位移不一致: {dict(deltas)}")
        for c in moved:
            if (c[2], c[3]) != main:
                d = next((dd for sid, dd in tagged if sid == c[1]), {})
                print(f"        反向 MOVED #{c[1]}: dx={c[2]:+d} dy={c[3]:+d}  "
                      f"color={cstr(d.get('color'))} {str(d.get('shape', ''))[:14]} "
                      f"size={d.get('size')} @({d.get('cx')},{d.get('cy')})")
    for c in gone:
        d = c[2]
        print(f"      GONE   #{c[1]} color={cstr(d['color'])} {d['shape'][:10]} size={d['size']} @({d['cx']},{d['cy']})")
    for c in appeared:
        d = c[2]
        print(f"      APPEAR #{c[1]} color={cstr(d['color'])} {d['shape'][:10]} size={d['size']} @({d['cx']},{d['cy']})")
    prev_grid = grid
    prev_n = n
    if done:
        print("  (done)")
        break

env.close()
