#!/usr/bin/env python3
"""验证空命令反馈改动: 各种情况下模型看到的 FEEDBACK 到底变没变、变成什么。

登录节点没有 torch, 无法 import 整个 codebase_rollout, 因此**直接解析真实源文件**,
把 _BASH_BLOCK_RE / _ANY_FENCE_RE / _EMPTY_CMD_NOTE_GENERIC / _empty_command_note
四个顶层定义原样取出执行 —— 测的是线上那份代码, 不是副本。

三部分:
  A. 接线检查   —— 调用点确实按开关分流, 且开关默认关(不影响在跑的实验)
  B. 用例表     —— 每种输入下, 开关关 / 开关开 时模型收到的完整 FEEDBACK 对比
  C. 真实轨迹回放 —— 拿 deltawin3 已发生的空命令重跑一遍, 看各类占比与文案变化

用法: python3 scripts/test_empty_command_note.py [--traj-rollouts rollout_104 rollout_110 ...]
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "codebase_rollout.py")
TRAJ = os.path.join(HERE, "logs/smith-4b-v3nocurr-deltawin3/traj/train")
# clbench 侧返回的观测(FEEDBACK 第一行), 本次改动不碰它。
OBS = "Empty command. Please provide a bash command."
WANT = ["_BASH_BLOCK_RE", "_ANY_FENCE_RE", "_EMPTY_CMD_NOTE_GENERIC", "_empty_command_note"]

fails: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'✓' if cond else '❌'} {label}")
    if not cond:
        fails.append(label)


def load_from_source() -> dict:
    """从真实源文件里取出被测的顶层定义并执行。"""
    src = open(SRC).read()
    tree = ast.parse(src)
    picked, ns = [], {"re": re}
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in WANT:
            picked.append(name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), SRC, "exec"), ns)
    missing = [w for w in WANT if w not in picked]
    if missing:
        sys.exit(f"❌ 源文件里找不到: {missing}")
    return ns


def feedback(note: str) -> str:
    """还原模型实际收到的那条消息(codebase_rollout.py 的 FEEDBACK 拼装)。"""
    return f"FEEDBACK: {OBS}\n{note}".strip()


# ── A. 接线检查 ───────────────────────────────────────────────────────────────
def part_a(src: str) -> None:
    print("\n【A】接线检查")
    check(
        'os.environ.get("CODEBASE_MULTIBLOCK_FEEDBACK", "0")' in src,
        "开关读 CODEBASE_MULTIBLOCK_FEEDBACK, 默认值为 '0'(默认关 ⇒ 在跑的实验行为不变)",
    )
    check(
        bool(re.search(r"_note = \(\s*_empty_command_note\(clean_act\)\s*if _MULTIBLOCK_FEEDBACK\s*else _EMPTY_CMD_NOTE_GENERIC", src)),
        "调用点按开关分流: 开=按成因诊断, 关=原文案",
    )
    for dead in ("_EMPTY_CMD_DIAGNOSTIC", "_empty_command_observation", "_EMPTY_CMD_GENERIC"):
        check(dead not in src, f"旧的重复实现 {dead} 已删干净(不会和新反馈叠加)")


# ── B. 用例表 ────────────────────────────────────────────────────────────────
CASES = [
    ("5 个 ```bash 块", "think\n```bash\nls\n```\n```bash\ncat a\n```\n```bash\npwd\n```\n```bash\nid\n```\n```bash\ndf\n```", "multi"),
    ("2 个 ```bash 块", "think\n```bash\nls\n```\nthen\n```bash\npwd\n```", "multi"),
    ("只有 ```python 块", "let me write it\n```python\nprint(1)\n```", "nonbash"),
    ("```json + ```text", "```json\n{}\n```\n```text\nhi\n```", "nonbash"),
    ("```bash 没闭合", "think\n```bash\nls -la", "generic"),
    ("无标签围栏 ```", "think\n```\nls\n```", "generic"),
    ("压根没有代码块", "I think the file is broken, let me reconsider.", "generic"),
    ("空字符串", "", "generic"),
]


def part_b(ns: dict) -> None:
    note_fn, generic = ns["_empty_command_note"], ns["_EMPTY_CMD_NOTE_GENERIC"]
    print("\n【B】各情况下模型收到的 FEEDBACK(开关关 → 开关开)")
    for label, text, kind in CASES:
        off, on = generic, note_fn(text)
        changed = on != off
        print(f"\n  ── {label} ──")
        print(f"     开关关: {feedback(off)}")
        print(f"     开关开: {feedback(on)}")
        if kind == "multi":
            n = len(ns["_BASH_BLOCK_RE"].findall(text))
            check(changed and f"contained {n} ```bash code blocks" in on, f"{label}: 变了, 且块数报对({n})")
            check("no command was executed" in on, f"{label}: 明说没有命令被执行")
        elif kind == "nonbash":
            check(changed and "not a ```bash block" in on, f"{label}: 变了, 且指出标签不对")
            check("only ```bash blocks are executed" in on, f"{label}: 说明只有 ```bash 会被执行")
        else:
            check(not changed, f"{label}: **不变**(原文案本来就准确, 不该动)")


# ── C. 真实轨迹回放 ──────────────────────────────────────────────────────────
def part_c(ns: dict, rollouts: list[str]) -> None:
    note_fn, generic = ns["_empty_command_note"], ns["_EMPTY_CMD_NOTE_GENERIC"]
    print(f"\n【C】真实轨迹回放(deltawin3 {', '.join(rollouts)})")
    tot, changed, kinds = 0, 0, {}
    samples: dict[str, str] = {}
    for rd in rollouts:
        for f in sorted(glob.glob(os.path.join(TRAJ, rd, "ep_*.json"))):
            for tr in json.load(open(f))["trials"]:
                for t in tr["turns"]:
                    if OBS not in (t.get("observation") or ""):
                        continue
                    tot += 1
                    act = t.get("assistant") or ""
                    on = note_fn(act)
                    if on == generic:
                        k = "沿用原文案(没闭合/没写块)"
                    elif "```bash code blocks" in on:
                        k = "改成: 你写了 N 个块, 只允许 1 个"
                    else:
                        k = "改成: 你的块不是 ```bash"
                    kinds[k] = kinds.get(k, 0) + 1
                    if on != generic:
                        changed += 1
                        samples.setdefault(k, on)
    if not tot:
        print("  ⚠️ 没找到轨迹, 跳过(不计入失败)")
        return
    print(f"  空命令共 {tot} 次, 反馈会变的 {changed} 次({100 * changed / tot:.1f}%)")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"    {v:5d}  {100 * v / tot:5.1f}%   {k}")
        if k in samples:
            print(f"           新文案: {samples[k]}")
    check(changed > 0, "真实数据上确实产生了变化")
    check(
        all("```bash" in note_fn(a) or True for a in [""]) and note_fn("") == generic,
        "边界: 空输入回落到原文案, 不抛异常",
    )


# ── D. 全量回放 + 健全性断言 ─────────────────────────────────────────────────
def part_d(ns: dict) -> None:
    """把 deltawin3 全部训练轨迹里的每一轮都过一遍, 逐条验证判据没有说错话。

    这里查的不是"分布好不好看", 而是四条**不能违反**的性质:
      P1 只在判空的轮次上产生文案(命令真的执行了的轮次一个都不能碰);
      P2 说"你写了 N 个块"时, N 必须等于实际完整块数, 且 N>=2;
      P3 说"你的块不是 ```bash"时, 实际完整 bash 块数必须为 0,
         且确实存在一个标签既非空也非 bash 的围栏(否则就是在撒谎);
      P4 其余一律逐字等于原文案。
    """
    note_fn, generic = ns["_empty_command_note"], ns["_EMPTY_CMD_NOTE_GENERIC"]
    BASH, FENCE = ns["_BASH_BLOCK_RE"], ns["_ANY_FENCE_RE"]
    dirs = sorted(glob.glob(os.path.join(TRAJ, "rollout_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    print(f"\n【D】全量回放({len(dirs)} 个 rollout)")
    if not dirs:
        print("  ⚠️ 没找到轨迹, 跳过")
        return

    turns = empties = 0
    kinds: dict[str, int] = {}
    v1 = v2 = v3 = v4 = 0  # 各性质的违例数
    ex: dict[str, str] = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "ep_*.json"))):
            try:
                data = json.load(open(f))
            except Exception:
                continue
            for tr in data["trials"]:
                for t in tr["turns"]:
                    turns += 1
                    act = t.get("assistant") or ""
                    is_empty = OBS in (t.get("observation") or "")
                    out = note_fn(act)
                    if not is_empty:
                        # P1: 命令执行了的轮次, 调用点根本不会走到这里; 但若有人误用,
                        # 至少要保证不会对一个成功执行的响应喊"没有命令被执行"。
                        if len(BASH.findall(act)) == 1 and out != generic:
                            v1 += 1
                            ex.setdefault("P1", f"{f}")
                        continue
                    empties += 1
                    n_bash = len(BASH.findall(act))
                    tags = [x.lower() for x in FENCE.findall(act)]
                    if "```bash code blocks" in out:
                        k = "① 多块"
                        m = re.search(r"contained (\d+) ```bash", out)
                        if not m or int(m.group(1)) != n_bash or n_bash < 2:
                            v2 += 1
                            ex.setdefault("P2", f"{f} (报 {m.group(1) if m else '?'}, 实际 {n_bash})")
                    elif "not a ```bash block" in out:
                        k = "② 非 bash 标签"
                        if n_bash != 0 or not any(x not in ("", "bash") for x in tags):
                            v3 += 1
                            ex.setdefault("P3", f"{f} (bash块={n_bash}, 标签={tags[:4]})")
                    else:
                        k = "③④ 沿用原文案"
                        if out != generic:
                            v4 += 1
                            ex.setdefault("P4", f"{f}")
                    kinds[k] = kinds.get(k, 0) + 1

    print(f"  扫过 {turns} 轮, 其中判空 {empties} 轮({100 * empties / max(turns, 1):.1f}%)")
    for k, v in sorted(kinds.items()):
        print(f"    {v:6d}  {100 * v / max(empties, 1):5.1f}%   {k}")
    ch = kinds.get("① 多块", 0) + kinds.get("② 非 bash 标签", 0)
    print(f"  反馈会改写的: {ch} / {empties}  ({100 * ch / max(empties, 1):.1f}%)")
    for name, bad, desc in (
        ("P1", v1, "没有对成功执行的响应误报"),
        ("P2", v2, "「你写了 N 个块」的 N 全部等于实际块数且 ≥2"),
        ("P3", v3, "「你的块不是 bash」只在确实没有 bash 块、且有非 bash 标签时才说"),
        ("P4", v4, "其余逐字等于原文案"),
    ):
        check(bad == 0, f"{name}: {desc}" + (f" —— {bad} 处违例, 例: {ex.get(name)}" if bad else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-rollouts", nargs="*", default=["rollout_104", "rollout_110", "rollout_116"])
    ap.add_argument("--full", action="store_true", help="额外跑 D: 全量轨迹回放 + 健全性断言")
    a = ap.parse_args()

    src = open(SRC).read()
    ns = load_from_source()
    print(f"被测源文件: {SRC}")
    part_a(src)
    part_b(ns)
    part_c(ns, a.traj_rollouts)
    if a.full:
        part_d(ns)

    print("\n" + "=" * 70)
    if fails:
        print(f"❌ {len(fails)} 项未通过:")
        for f in fails:
            print(f"   - {f}")
        sys.exit(1)
    print("✅ 全部通过")


if __name__ == "__main__":
    main()
