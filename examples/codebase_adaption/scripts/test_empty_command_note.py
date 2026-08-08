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
WANT = [
    "_BASH_BLOCK_RE",
    "_ANY_FENCE_RE",
    "_EMPTY_CMD_NOTE_GENERIC",
    "_empty_command_note",
]

fails: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'✓' if cond else '❌'} {label}")
    if not cond:
        fails.append(label)


def load_from_source() -> dict:
    """从真实源文件里取出被测的顶层定义并执行。"""
    src = open(SRC).read()
    tree = ast.parse(src)
    picked, ns = [], {"re": re, "os": os}
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
    # 旧写法是 all(... or True ...), 恒真 —— 只测了"不抛异常"这一半(FLAME review 2026-08-08)。
    check(note_fn("") == generic, "边界: 空字符串回落到原文案")
    check(note_fn(None) == generic, "边界: None 回落到原文案, 不抛异常")


# ── D. 全量回放 + 健全性断言 ─────────────────────────────────────────────────
def part_d(ns: dict) -> None:
    """把 deltawin3 全部训练轨迹里的每一轮都过一遍, 逐条验证判据没有说错话。

    这里查的不是"分布好不好看", 而是四条**不能违反**的性质:
      P1 只在判空的轮次上产生文案(命令真的执行了的轮次一个都不能碰);
      P2 说"你写了 N 个块"时, N 必须等于实际完整块数, 且 N>=2;
      P3 说"你的块不是 ```bash"时, 实际完整 bash 块数必须为 0,
         且确实存在一个标签既非空也非 bash 的围栏(否则就是在撒谎);
      P4 其余一律逐字等于原文案;
      P5 "块是空的"只在恰好 1 个块且内容为空白时才说。
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
    v1 = v2 = v3 = v4 = v5 = 0  # 各性质的违例数
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
                        # P1: 命令真的执行了的轮次, 绝不能收到"所以没有命令被执行"这类文案。
                        # (旧版本这里加了 len(blocks)==1 的前置条件, 使断言恒真、永远发现不了
                        #  问题; FLAME review 2026-08-08 指出。)
                        if out != generic:
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
                    elif "empty ```bash block" in out:
                        k = "③ 块存在但为空"
                        blk = BASH.findall(act)
                        if n_bash != 1 or blk[0].strip():
                            v5 += 1
                            ex.setdefault("P5", f"{f} (bash块={n_bash})")
                    else:
                        k = "④ 沿用原文案"
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
        ("P5", v5, "「块是空的」只在恰好 1 个块且内容为空白时才说"),
    ):
        check(bad == 0, f"{name}: {desc}" + (f" —— {bad} 处违例, 例: {ex.get(name)}" if bad else ""))


# ── E. 空块 / 与解析器判定保持一致 ────────────────────────────────────────────
def part_e(ns: dict) -> None:
    """FLAME review 2026-08-08 的两类"说假话"场景, 外加一条防跨分支复制的护栏。

    E1 块找到了但内容为空 —— 不能说"没找到完整的块"。
    E2 文案必须与本文件 _response_from_text 的判定一致。上一版从 orchard 分支抄了
       CODEBASE_MULTIBLOCK_TAKE_FIRST 来抑制多块文案, 但本分支解析器不看那个变量,
       开关一开反而把准确文案换成假话。这里直接盯住解析器源码里的判定规则:
       规则一变(比如合并了 orchard 的 take-first), 测试立刻失败, 强制重新对齐文案。
    E3 "空块 + 后面还有块"不进空命令路径 —— 正则回溯会并成 1 个内容为 ``` 的块,
       解析器会真的去执行 ```。这是解析器层面的既有问题, 这里断言其真实后果, 免得
       被"不可达"三个字轻轻带过。
    """
    print("\n【E】空块 / 与解析器判定一致")
    note_fn, generic = ns["_empty_command_note"], ns["_EMPTY_CMD_NOTE_GENERIC"]
    src = open(SRC).read()

    for lab, txt in (
        ("单个空块", "Let me run it.\n\n```bash\n\n```"),
        ("单个只有空白的块", "Let me run it.\n\n```bash\n   \n```"),
    ):
        out = note_fn(txt)
        print(f"  {lab}: {out}")
        check(out != generic, f"E1 {lab}: 不再回落到「没找到完整的块」")
        check("empty ```bash block" in out, f"E1 {lab}: 明说块是空的")

    # E2: 解析器规则 = 恰好 1 块才执行; 文案里的多块分支正是建立在这条规则上。
    check(
        'command = blocks[0].strip() if len(blocks) == 1 else ""' in src,
        "E2: 解析器仍是「恰好 1 块才执行」—— 规则若变, 多块文案需重新评估",
    )
    # 用 AST 查真实的名字引用, 不看注释/docstring —— 否则解释"为何撤除"的那段文字
    # 自己就会把断言匹配掉。
    _fn = next(
        n for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "_empty_command_note"
    )
    _names = {n.id for n in ast.walk(_fn) if isinstance(n, ast.Name)}
    check(
        not any("TAKE_FIRST" in x for x in _names),
        f"E2: 文案函数代码里不再引用 TAKE_FIRST 类开关(实际引用: {sorted(_names)})",
    )
    out = note_fn("Two blocks.\n\n```bash\nls\n```\n\n```bash\npytest\n```")
    check("2 ```bash code blocks" in out, "E2: 多块仍如实报块数(不受任何环境变量抑制)")

    # E3: 空块后面还有块 → 解析器把 command 设成 ```, 真的会去执行, 不走空命令路径。
    B = ns["_BASH_BLOCK_RE"]
    for t in ("```bash\n\n```\n\n```bash\npytest\n```", "```bash\n   \n```\n```bash\npytest\n```"):
        blk = B.findall(t)
        cmd = blk[0].strip() if len(blk) == 1 else ""
        check(
            len(blk) == 1 and cmd == "```",
            f"E3: 「空块+后续块」被并成 1 块, 解析出的命令是 {cmd!r}(会被真的执行, 属解析器既有问题)",
        )


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
    part_e(ns)
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
