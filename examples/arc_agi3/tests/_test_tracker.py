import os
import sys

sys.path.insert(0, "/home/qixinx/ARC-AGI-3-Agents")
sys.path.insert(0, "/home/qixinx/miles")
import arc_rl.arc_env  # noqa: F401  (triggers load_dotenv)

if os.environ.get("ARC_BASE_URL"):
    os.environ["ARC_BASE_URL"] = os.environ["ARC_BASE_URL"].rstrip("/")
from arc_rl.arc_env import ArcEnv
from examples.arc_agi3.grid_render import objects_diff_text
from examples.arc_agi3.object_tracker import ObjectTracker


def cstr(c):
    return "/".join(map(str, c)) if isinstance(c, list) else str(c)


env = ArcEnv("ar25-0c556536", max_actions=12)
obs = env.reset()
tracker = ObjectTracker()
tagged, bg, changes = tracker.update(obs["grid"])

print(f"=== reset: {len(tagged)} objects (bg={bg}) — 初始 id 分配 ===")
for sid, d in sorted(tagged):
    print(f"  #{sid}: color={cstr(d['color'])} shape={d['shape'][:22]} center=({d['cx']},{d['cy']})")

trajectory = {sid: [(d["cx"], d["cy"])] for sid, d in tagged}
id_color = {sid: d["color"] for sid, d in tagged}
prev_grid = obs["grid"]

for step_i in range(6):
    avail = obs.get("available") or []
    if not avail:
        break
    act = "ACTION1" if 1 in avail else f"ACTION{avail[0]}"
    obs, r, done, info = env.step(act)
    grid = obs["grid"]
    if grid == prev_grid:
        print(f"\n[step {step_i + 1} {act}] 屏幕无变化, skip")
        prev_grid = grid
        if done:
            break
        continue
    tagged, bg, changes = tracker.update(grid)
    for sid, d in tagged:
        trajectory.setdefault(sid, []).append((d["cx"], d["cy"]))
        id_color.setdefault(sid, d["color"])

    print(f"\n=== after {act} (step {step_i + 1}) ===")
    print("  TRACKER (Hungarian) changes:")
    for ch in changes:
        if ch[0] == "MOVED":
            print(f"    MOVED   #{ch[1]} (color={cstr(id_color.get(ch[1]))}): dx={ch[2]:+d} dy={ch[3]:+d}")
        elif ch[0] == "RESHAPED":
            print(f"    RESHAPED #{ch[1]} (color={cstr(id_color.get(ch[1]))}): {ch[2][:14]} -> {ch[3][:14]}  dx={ch[4]:+d} dy={ch[5]:+d}")
        elif ch[0] == "APPEARED":
            print(f"    APPEARED #{ch[1]} (color={cstr(ch[2]['color'])} @({ch[2]['cx']},{ch[2]['cy']}))")
        elif ch[0] == "GONE":
            print(f"    GONE    #{ch[1]} (color={cstr(ch[2]['color'])} @({ch[2]['cx']},{ch[2]['cy']}))")

    greedy = objects_diff_text(prev_grid, grid)
    dot_lines = [ln for ln in greedy.split("\n") if "color=0" in ln]
    print("  GREEDY objects_diff_text — color=0 dots (现状, 看有没有 dy=-9 伪影):")
    for ln in dot_lines:
        print(f"    {ln}")
    prev_grid = grid
    if done:
        break

print("\n=== 验证: 每个 stable id 的 center 轨迹 + 逐帧位移 ===")
ok_stable = True
for sid in sorted(trajectory):
    traj = trajectory[sid]
    if len(traj) < 2:
        continue
    deltas = [(traj[i + 1][0] - traj[i][0], traj[i + 1][1] - traj[i][1]) for i in range(len(traj) - 1)]
    tag = "  <- color=0 dot" if id_color.get(sid) == 0 else ""
    print(f"  #{sid} (color={cstr(id_color.get(sid))}): {traj}  Δ={deltas}{tag}")

# 量化判定: color=0 dot 的逐帧位移是否全一致 (无 dy=-9 串位)
print("\n=== 判定 ===")
dot_deltas = []
for sid in trajectory:
    if id_color.get(sid) == 0:
        traj = trajectory[sid]
        dot_deltas += [(traj[i + 1][0] - traj[i][0], traj[i + 1][1] - traj[i][1]) for i in range(len(traj) - 1)]
uniq = set(dot_deltas)
print(f"  color=0 dot 的所有逐帧位移集合 = {uniq}")
print(f"  -> {'✅ 全部一致 (Hungarian 消除了 dot 串位伪影)' if len(uniq) <= 1 else '⚠ 仍有不一致位移: ' + str(uniq)}")
env.close()
