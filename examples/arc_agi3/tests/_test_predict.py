import os
import sys

sys.path.insert(0, "/home/qixinx/ARC-AGI-3-Agents")
sys.path.insert(0, "/home/qixinx/miles")
import arc_rl.arc_env  # noqa: F401

if os.environ.get("ARC_BASE_URL"):
    os.environ["ARC_BASE_URL"] = os.environ["ARC_BASE_URL"].rstrip("/")
from arc_rl.arc_env import ArcEnv
from examples.arc_agi3.object_tracker import ObjectTracker


def scenario(label, action, steps):
    print(f"\n{'=' * 64}\n场景: {label}  (action={action} ×{steps})\n{'=' * 64}")
    env = ArcEnv("ar25-0c556536", max_actions=steps + 4)
    obs = env.reset()
    tr = ObjectTracker()
    tagged, bg, _ = tr.update(obs["grid"])
    c4_id = next((tid for tid, d, st in tagged if d["color"] == 4 and d["size"] > 20), None)
    dot_ids = {tid for tid, d, st in tagged if d["color"] == 0}
    print(f"reset: {len(tagged)} objs | 追踪 color4-mask=#{c4_id} | dots={sorted(dot_ids)}")
    hist = {}
    for tid, d, st in tagged:
        hist.setdefault(tid, []).append((d["cx"], d["cy"], st))
    prev = obs["grid"]
    for i in range(steps):
        if int(action[-1]) not in (obs.get("available") or []):
            print(f"[{i + 1}] {action} 不可用")
            break
        obs, r, done, info = env.step(action)
        g = obs["grid"]
        if g == prev:
            print(f"[{i + 1}] {action} 屏幕无变化")
            prev = g
            if done:
                break
            continue
        tagged, bg, changes = tr.update(g)
        for tid, d, st in tagged:
            hist.setdefault(tid, []).append((d["cx"], d["cy"], st))
        kinds = {k: [c[1] for c in changes if c[0] == k]
                 for k in ("OCCLUDED", "REAPPEARED", "GONE", "APPEARED")}
        ex = "  ".join(f"{k}={v}" for k, v in kinds.items() if v)
        c4_now = next((f"#{tid}/{st}" for tid, d, st in tagged if tid == c4_id), "LOST")
        print(f"[{i + 1}] {action} objs={len(tagged)}  c4mask={c4_now}  {ex}")
        prev = g
        if done:
            break
    env.close()
    return c4_id, dot_ids, hist


# ---------- 场景1: 普通移动 ----------
c4_1, dots1, h1 = scenario("普通移动(一路上推)", "ACTION1", 5)
dot_deltas = set()
dot_occluded = False
for tid in dots1:
    traj = h1.get(tid, [])
    for j in range(len(traj) - 1):
        if traj[j][2] == "visible" and traj[j + 1][2] == "visible":
            dot_deltas.add((traj[j + 1][0] - traj[j][0], traj[j + 1][1] - traj[j][1]))
        if traj[j + 1][2] == "occluded":
            dot_occluded = True
print(f"\n  [场景1判定] dots 可见时位移集合={dot_deltas}  误触发occluded={dot_occluded}")
print(f"  -> {'✅ 普通移动: dots 稳定追踪、全一致、无误判遮挡' if dot_deltas <= {(0, -3)} and not dot_occluded else '⚠ 异常'}")

# ---------- 场景2: 镜像穿越遮挡 ----------
c4_2, dots2, h2 = scenario("镜像穿越遮挡(一路右推)", "ACTION4", 8)
print(f"\n  [场景2判定] color4-mask 初始 id=#{c4_2} 的完整轨迹:")
traj = h2.get(c4_2, [])
print("    " + " -> ".join(f"({x},{y},{s[:3]})" for x, y, s in traj))
n_occ = sum(1 for _, _, s in traj if s == "occluded")
n_vis_after = sum(1 for _, _, s in traj if s == "visible")
print(f"    存活 {len(traj)} 帧, occluded {n_occ} 帧, visible {n_vis_after} 帧")
ok = len(traj) >= 4 and n_occ >= 1 and traj[-1][2] == "visible"
print(f"  -> {'✅ #%d 穿越遮挡后 id 未断裂 (occluded→reappeared, 始终同一个 id)' % c4_2 if ok else '⚠ id 可能仍断裂或未触发遮挡, 看上面轨迹'}")
