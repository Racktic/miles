"""Prompts for codebase-adaptation memory co-training.

Mirrors the offline clbench ``icl_summary`` (replace) system so miles' ACT/WRITE
prompts match the official scaffold
(continual-learning-bench/src/systems/icl_summary/system.py):

  - ACT: **no** system message; the running memory is injected on the first turn
    of each issue as ``[Memory of what you've learned so far]``; command output
    is fed back as a separate ``FEEDBACK:`` user message (done by the caller, as
    the offline system does via ``observe``); the JSON schema is **not** inlined
    into the prompt — structure is enforced by sglang json_schema constrained
    decoding, mirroring the offline ``response_format`` path. The task's own
    ``_SYSTEM_TEMPLATE`` (thought/command format spec) already rides inside
    ``query.prompt``, so no extra format instruction is needed.
  - WRITE: ``MEMORY_SYSTEM`` + the finished issue's raw turns + a final
    instruction that rewrites the memory into the two-section format
    (``### Repository Knowledge`` / ``### Lessons & Pitfalls``).

The wording is inherited verbatim from the offline system (whose class is named
``icl_summary``), but here we call it *memory* throughout — it is the running,
co-evolved memory store, not a one-shot summary.
"""

from __future__ import annotations


# ── WRITE (memory rewrite) — text verbatim from the offline icl_summary system ──
MEMORY_SYSTEM = (
    "You are a software engineer solving a sequence of coding tasks in shared "
    "repositories. You keep a running, structured MEMORY of what you have "
    "learned about these repositories and how to work in them effectively. At "
    "the end of each task you are given your PREVIOUS memory plus the task you "
    "just worked on, and you output the UPDATED memory — carrying forward what "
    "you already confirmed and folding in the new task's evidence — for use in "
    "future tasks."
)

MEMORY_INSTRUCTION = (
    "The task above is over. Update your running memory in EXACTLY these two "
    "sections:\n\n"
    "### Repository Knowledge\n"
    "One bullet per confirmed fact about each repository you have worked in: "
    "directory layout and key files, how to run the tests reliably, coding "
    "conventions, and where specific functionality lives.\n\n"
    "### Lessons & Pitfalls\n"
    "One bullet per lesson about HOW to work: approaches that succeeded, "
    "approaches that failed and why (e.g. fragile edit patterns, wrong test "
    "invocations, submitting without running the full test suite), and checks "
    "worth doing before submitting.\n\n"
    "Carry forward everything you already confirmed in your previous memory; "
    "ADD this task's new findings and only revise an entry when THIS task gives "
    "direct evidence against it — do NOT delete a confirmed fact just because a "
    "task went badly. Output ONLY the two sections."
)


def build_act_user_content(
    *,
    memory: str,
    query_prompt: str,
    include_memory: bool,
) -> str:
    """User content for an ACT turn, mirroring ``icl_summary.respond`` query_parts.

    The first turn of an issue injects the running memory; every turn ends with
    the task query. Feedback (command output) is a separate ``FEEDBACK:`` user
    message added by the caller — not merged here — exactly as the offline system
    does via ``observe``.
    """
    parts: list[str] = []
    if include_memory and memory and memory.strip():
        parts.append("[Memory of what you've learned so far]\n" + memory.strip())
    if query_prompt:
        parts.append(query_prompt)
    return "\n\n".join(parts) if parts else "(no content)"


# ── WRITE prompt v3(2026-07-19, 用户逐字审定; 离线 A/B: 串台 12-16/30 → 2-4/30,
#    严格合规 16-17/30 → 23-24/30): 三明治框架语 + 成败反思引导。
#    经 CODEBASE_WRITE_PROMPT_V3=1 启用; 默认关闭, 保证现役 run 重启口径不漂移。──
WRITE_TRANSCRIPT_PREAMBLE = (
    "Below is the complete, read-only transcript of the task you just finished. "
    "It is provided for reference only."
)

MEMORY_INSTRUCTION_V3_HEAD = (
    "Above is the entire transcript - the task is OVER. Do NOT continue it, do NOT "
    "run commands, do NOT output bash. Your ONLY job is to rewrite the memory based "
    "on the transcript.\n\n"
    "Reflect on the final outcome shown in the transcript before writing: if the "
    "submission PASSED, identify what was key to this success so it can be reused "
    "in future tasks; if it FAILED or the step budget ran out, diagnose the most "
    "likely root cause of the failure and record it as a lesson.\n\n"
)


def build_write_messages(
    *,
    previous_memory: str,
    instance_messages: list[dict[str, str]],
    v3: bool = False,
) -> list[dict[str, str]]:
    """WRITE messages, mirroring the offline memory-rewrite call (window=1):

    ``MEMORY_SYSTEM`` + the finished issue's raw turns + a final user turn
    carrying the previous memory and the two-section rewrite instruction.
    v3=True additionally wraps the transcript in the sandwich framing and
    prepends the outcome-reflection guidance (see WRITE prompt v3 above).
    """
    instr = MEMORY_INSTRUCTION
    if v3:
        instr = instr.replace(
            "The task above is over. Update your running memory",
            "Update your running memory",
            1,
        )
    prev = (previous_memory or "").strip()
    if prev:
        instr = f"Your previous memory:\n{prev}\n\n{instr}"
    if v3:
        instr = MEMORY_INSTRUCTION_V3_HEAD + instr
    turns = [dict(m) for m in instance_messages]
    pre = [{"role": "user", "content": WRITE_TRANSCRIPT_PREAMBLE}] if v3 else []
    return [
        {"role": "system", "content": MEMORY_SYSTEM},
        *pre,
        *turns,
        {"role": "user", "content": instr},
    ]


