import sys, os
sys.path.insert(0, "/home/qixinx/ARC-AGI-3-Agents")
sys.path.insert(0, "/home/qixinx/miles")
sys.path.insert(0, "/data/user_data/qixinx")  # arc_dsl_llm
import arc_rl.arc_env  # triggers load_dotenv
if os.environ.get("ARC_BASE_URL"):
    os.environ["ARC_BASE_URL"] = os.environ["ARC_BASE_URL"].rstrip("/")
from arc_rl.arc_env import ArcEnv
from examples.arc_agi3.grid_render import grid_to_objects_text, objects_diff_text, _segment
from arc_dsl_llm.dsl import as_objects

env = ArcEnv("ar25-0c556536", max_actions=4)
obs = env.reset()
before = obs["grid"]

# (1) 正确性：我的 _segment vs arc-dsl as_objects (single_color, 8-conn, discard_bg)
g_t = tuple(tuple(int(v) for v in row) for row in before)
mine, bg = _segment(before, single_color=True, diagonal=True)
theirs = as_objects(g_t, True, True, True)
mine_sizes = sorted(len(o) for o in mine)
their_sizes = sorted(len(o) for o in theirs)
print("_segment vs arc-dsl as_objects: %d vs %d objs, sizes match=%s (bg=%s)"
      % (len(mine), len(theirs), mine_sizes == their_sizes, bg))

# (2) s_t = object text（带 shape）
print("\n===== s_t = grid_to_objects_text(before) =====")
s_t = grid_to_objects_text(before)
print(s_t)
print("[%d chars ~ %d tokens  vs raw matrix ~4246]" % (len(s_t), len(s_t) // 3))

# (3) d_t = 物体级 diff（step 一个方向键拿 after）
avail = obs["available"]
act = "ACTION1" if 1 in avail else ("ACTION%d" % avail[0])
obs2, rew, done, info = env.step(act)
after = obs2["grid"]
print("\n===== d_t = objects_diff_text(before, after) after %s, changed=%s =====" % (act, before != after))
print(objects_diff_text(before, after))
env.close()
