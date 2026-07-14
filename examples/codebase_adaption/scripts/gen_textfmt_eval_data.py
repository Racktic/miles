#!/usr/bin/env python3
"""生成新 scaffold(纯文本格式)测评用的三份 eval jsonl(2026-07-13, 用户批准的计划)。

产出(写到 examples/codebase_adaption/data/):
  1. baseline_textfmt_eval.jsonl  — 259 行单题 episode(swecl 240 行 split=train,
     codebase 19 行 split=heldout;均 no_memory=true)。配 N_EVAL_SAMPLES=4 得 avg@4。
  2. icl_textfmt_eval.jsonl       — 5 行 9+10 episode,题序 = 官方 clbench 5-run 原样,
     no_memory=true(纯 ICL,全历史)。
  3. replace_textfmt_eval.jsonl   — 同 5 题序,不带 no_memory(= memory-replace 模式)。

题序来源: 官方 clbench icl 轨迹(replace 组), 并与 augment 组逐题 assert 一致(确定性证明)。
用法: python3 gen_textfmt_eval_data.py [--small]   (--small: baseline 只取前 10 题, 冒烟用)
"""
import argparse
import json
from pathlib import Path

MILES_DATA = Path("/home/qixinx/miles/examples/codebase_adaption/data")
CLBENCH = Path("/home/qixinx/continual-learning-bench")
TRACE_REPLACE = CLBENCH / "results/codebase_adaptation/traces/2026-07-12T16-25-00.746359Z"
TRACE_AUGMENT = CLBENCH / "results/codebase_adaptation/traces/2026-07-12T16-25-00.763712Z"
FULL_JSONL = CLBENCH / "data/swe_bench_cl/full.jsonl"


def stage_label(instance_id: str) -> str:
    if instance_id.startswith("jazzband__tablib"):
        return "tablib"
    if instance_id.startswith("jd__tenacity"):
        return "tenacity"
    return instance_id.split("__")[0]


def episode_line(text: str, split: str, instance_ids: list, *, no_memory: bool, order_rank=None) -> str:
    meta = {
        "split": split,
        "instance_ids": instance_ids,
        "stage_labels": [stage_label(i) for i in instance_ids],
    }
    if no_memory:
        meta["no_memory"] = True
    if order_rank is not None:
        meta["order_rank"] = order_rank
    return json.dumps({
        "prompt": [{"role": "user", "content": [{"type": "text", "text": text}]}],
        "metadata": meta,
    })


def official_orders() -> list:
    """官方 clbench 5-run 的题序, 并用两组独立轨迹交叉 assert 确定性。"""
    def orders(group: Path) -> list:
        out = []
        for i in range(5):
            d = json.loads((group / f"run_{i:04d}.json").read_text())
            out.append([o["instance_id"] for o in d["instance_outcomes"]])
        return out

    rep, aug = orders(TRACE_REPLACE), orders(TRACE_AUGMENT)
    assert rep == aug, "两组官方轨迹题序不一致, permute 假设被打破!"
    assert all(len(s) == 19 for s in rep)
    return rep


def gen_patch() -> None:
    """补测文件: 扫描 bl-4b-textfmt 已落盘轨迹, 找出不满 4 样本的题, 每题 1 行(x4 由
    N_EVAL_SAMPLES 展开)。用途: l5-16 上 sympy-20428 镜像 GLIBC 不兼容炸掉 93% 进度的
    job 后, 在 v5-16(apptainer 已验证)补齐缺口(2026-07-13)。"""
    from collections import defaultdict

    per = defaultdict(int)
    traj = MILES_DATA.parent / "logs/bl-4b-textfmt/traj/eval/heldout/rollout_0"
    for f in traj.glob("ep_*.json"):
        d = json.loads(f.read_text())
        for tr in d.get("trials", []):
            per[tr["outcome"]["instance_id"]] += 1
    all_ids = set()
    for line in FULL_JSONL.read_text().splitlines():
        if line.strip():
            all_ids.add(json.loads(line)["instance_id"])
    all_ids |= set(official_orders()[0])
    need = sorted(i for i in all_ids if per.get(i, 0) < 4)
    lines = []
    for iid in need:
        split = "heldout" if iid.startswith(("jazzband", "jd__")) else "train"
        lines.append(episode_line(f"Baseline patch episode: {iid}", split, [iid],
                                  no_memory=True, order_rank=len(lines)))
    out = MILES_DATA / "baseline_textfmt_eval_patch.jsonl"
    out.write_text("\n".join(lines) + "\n")
    print(f"{out.name}: {len(lines)} 题(不满4样本者全部重测x4)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="baseline 只取前 10 题(冒烟)")
    ap.add_argument("--patch", action="store_true", help="生成补测文件(缺样本的题)")
    args = ap.parse_args()
    if args.patch:
        gen_patch()
        return

    # ── 1) baseline: 单题 episode ──
    swecl_ids = []
    for line in FULL_JSONL.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            swecl_ids.append(d["instance_id"])
    assert len(swecl_ids) == 240, f"swecl 池应为 240, 实际 {len(swecl_ids)}"

    orders = official_orders()
    codebase_ids = sorted(set(orders[0]))  # 19 题(任一 run 的题集合都相同)
    assert len(codebase_ids) == 19

    lines = []
    # order_rank 必填: eval 样本 group_index 为 None, rollout.py:557 的 shuffle_seed
    # 兜底会 int(None) 崩(冒烟实测)。单题 episode 的 rank 只影响 episode_id 命名。
    for iid in swecl_ids:
        lines.append(episode_line(f"Baseline single-issue episode: {iid}", "train", [iid],
                                  no_memory=True, order_rank=len(lines)))
    for iid in codebase_ids:
        lines.append(episode_line(f"Baseline single-issue episode: {iid}", "heldout", [iid],
                                  no_memory=True, order_rank=len(lines)))
    if args.small:
        lines = lines[:10]
    out = MILES_DATA / ("baseline_textfmt_eval_small.jsonl" if args.small else "baseline_textfmt_eval.jsonl")
    out.write_text("\n".join(lines) + "\n")
    print(f"{out.name}: {len(lines)} 行")

    if args.small:
        return

    # ── 2) icl(no_memory)与 3) replace(memory): 官方 5 题序 ──
    icl_lines, rep_lines = [], []
    for rank, seq in enumerate(orders):
        icl_lines.append(episode_line(f"ICL eval episode, official run {rank} order.", "heldout", seq, no_memory=True, order_rank=rank))
        rep_lines.append(episode_line(f"Memory-replace eval episode, official run {rank} order.", "heldout", seq, no_memory=False, order_rank=rank))
    (MILES_DATA / "icl_textfmt_eval.jsonl").write_text("\n".join(icl_lines) + "\n")
    (MILES_DATA / "replace_textfmt_eval.jsonl").write_text("\n".join(rep_lines) + "\n")
    print(f"icl_textfmt_eval.jsonl: {len(icl_lines)} 行(官方题序, no_memory)")
    print(f"replace_textfmt_eval.jsonl: {len(rep_lines)} 行(官方题序, memory-replace)")

    # ── 覆盖自检 ──
    all_ids = set(swecl_ids) | set(codebase_ids)
    print(f"覆盖自检: 共 {len(all_ids)} 题(应 259)", "OK" if len(all_ids) == 259 else "!!")


if __name__ == "__main__":
    main()
