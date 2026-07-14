#!/usr/bin/env python3
"""汇总新 scaffold 测评结果(2026-07-13, 用户批准计划)。

用法:
  python3 aggregate_textfmt_eval.py baseline        # -> data/baseline_4b_textfmt.json + 对比报告
  python3 aggregate_textfmt_eval.py baseline_small  # 冒烟自检(不写 baseline json)
  python3 aggregate_textfmt_eval.py icl
  python3 aggregate_textfmt_eval.py replace
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

SD = Path("/home/qixinx/miles/examples/codebase_adaption")
RUN_IDS = {
    "baseline": "bl-4b-textfmt",
    "baseline_small": "bl-4b-textfmt-small",
    "icl": "icl-4b-textfmt",
    "replace": "iclmem-4b-textfmt",
}


def load_episodes(run_id: str) -> list:
    traj = SD / "logs" / run_id / "traj" / "eval" / "heldout" / "rollout_0"
    files = sorted(traj.glob("ep_*.json"))
    assert files, f"没有轨迹: {traj}"
    return [json.loads(f.read_text()) for f in files]


def agg_baseline(suite: str) -> None:
    eps = load_episodes(RUN_IDS[suite])
    per_inst = defaultdict(list)
    for ep in eps:
        for tr in ep.get("trials", []):
            o = tr["outcome"]
            per_inst[o["instance_id"]].append(float(o.get("reward") or 0.0))

    # 合并补测(bl-4b-textfmt-patch): 补测题以补测的 4 个新样本为准(整题替换, 口径均一)
    if suite == "baseline":
        patch_dir = SD / "logs" / "bl-4b-textfmt-patch" / "traj" / "eval" / "heldout" / "rollout_0"
        if patch_dir.exists():
            patched = defaultdict(list)
            for f in sorted(patch_dir.glob("ep_*.json")):
                d = json.loads(f.read_text())
                for tr in d.get("trials", []):
                    o = tr["outcome"]
                    patched[o["instance_id"]].append(float(o.get("reward") or 0.0))
            for iid, vals in patched.items():
                per_inst[iid] = vals
            print(f"(已合并补测 {len(patched)} 题, 整题替换)")

    n_expected = 10 if suite == "baseline_small" else 259
    n_samples = 2 if suite == "baseline_small" else 4
    print(f"覆盖: {len(per_inst)} 题(应 {n_expected});样本数分布: "
          f"{sorted(set(len(v) for v in per_inst.values()))}(应 [{n_samples}])")

    outcomes = [
        {"instance_id": iid, "reward": round(sum(v) / len(v), 4)}
        for iid, v in sorted(per_inst.items())
    ]
    solved_any = sum(1 for v in per_inst.values() if any(x > 0 for x in v))
    mean_r = sum(o["reward"] for o in outcomes) / len(outcomes)
    print(f"avg@{n_samples} 均值: {mean_r:.4f};pass@{n_samples}(任一次>0): {solved_any}/{len(per_inst)}")

    if suite == "baseline":
        out = SD / "data" / "baseline_4b_textfmt.json"
        out.write_text(json.dumps({"instance_outcomes": outcomes}, indent=2))
        print(f"写出: {out}")
        # 与旧 baseline 对比
        old = {o["instance_id"]: o["reward"]
               for o in json.loads((SD / "data" / "baseline_4b_merged.json").read_text())["instance_outcomes"]}
        pairs = [(o["reward"], old[o["instance_id"]]) for o in outcomes if o["instance_id"] in old]
        new_m = sum(p[0] for p in pairs) / len(pairs)
        old_m = sum(p[1] for p in pairs) / len(pairs)
        n = len(pairs)
        mx = sum(p[0] for p in pairs) / n
        my = sum(p[1] for p in pairs) / n
        cov = sum((a - mx) * (b - my) for a, b in pairs) / n
        vx = sum((a - mx) ** 2 for a, b in pairs) / n
        vy = sum((b - my) ** 2 for a, b in pairs) / n
        corr = cov / ((vx * vy) ** 0.5 + 1e-12)
        print(f"新旧对比(n={n}): 新均值 {new_m:.4f} vs 旧均值 {old_m:.4f}(漂移 {new_m-old_m:+.4f});相关 r={corr:.3f}")


def agg_icl(suite: str) -> None:
    eps = load_episodes(RUN_IDS[suite])
    bl_path = SD / "data" / "baseline_4b_textfmt.json"
    if not bl_path.exists():
        bl_path = SD / "data" / "baseline_4b_merged.json"
        print(f"(新 baseline 未就绪, gain 暂用旧口径: {bl_path.name})")
    bl = {o["instance_id"]: o["reward"] for o in json.loads(bl_path.read_text())["instance_outcomes"]}

    scores, gains = [], []
    print(f"{'run':>4} {'score(均reward)':>14} {'cum_gain':>9} {'solved':>7}")
    for ep in sorted(eps, key=lambda e: e.get("order_rank") or 0):
        outs = [tr["outcome"] for tr in ep.get("trials", [])]
        rs = [float(o.get("reward") or 0.0) for o in outs]
        g = sum(r - bl.get(o["instance_id"], 0.0) for r, o in zip(rs, outs))
        sc = sum(rs) / len(rs)
        scores.append(sc); gains.append(g)
        rank = ep.get("order_rank")
        print(f"{str(rank):>4} {sc:>14.4f} {g:>+9.4f} {sum(1 for o in outs if o.get('success')):>4}/19")
    print(f"\n5-run 均值 score={sum(scores)/len(scores):.4f}  cum_gain={sum(gains)/len(gains):+.4f}")
    ref = {"icl": 0.0347, "replace": 0.0706}[suite]
    print(f"旧 scaffold 离线参考({suite}): {ref}")


if __name__ == "__main__":
    suite = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if suite.startswith("baseline"):
        agg_baseline(suite)
    else:
        agg_icl(suite)
