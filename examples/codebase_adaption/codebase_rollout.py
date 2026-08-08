"""Miles rollout for codebase-adaptation test-time memory co-training."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import truncate_samples_by_total_tokens
from miles.utils.http_utils import post
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample

from examples.codebase_adaption.codebase_advantage import downstream_improve_rewards
from examples.codebase_adaption.prompts import (
    build_act_user_content,
    build_write_messages,
)
from examples.codebase_adaption.schedule import EpisodeOrder, build_episode_order, load_baseline_rewards, stage_lookup_for


_MASK_GEN = None
_BASELINE_CACHE: dict[str, float] | None = None


class _QwenMaskGen(MultiTurnLossMaskGenerator):
    """Multi-turn assistant-token loss mask with Qwen empty-think cleanup."""

    _think_seqs = None

    def get_loss_mask(self, messages, tools=None):
        ids, mask = super().get_loss_mask(messages, tools=tools)
        if self._think_seqs is None:
            self._think_seqs = [
                self.tokenizer(s, add_special_tokens=False)["input_ids"]
                for s in ("<think>\n\n</think>\n\n", "<think>\n\n</think>")
            ]
        for sub in self._think_seqs:
            length = len(sub)
            if not length:
                continue
            i = 0
            while i <= len(ids) - length:
                if ids[i : i + length] == sub:
                    for j in range(i, i + length):
                        mask[j] = 0
                    i += length
                else:
                    i += 1
        return ids, mask


def _clbench_root(args) -> Path:
    root = os.environ.get("CLBENCH_ROOT") or getattr(args, "codebase_clbench_root", None)
    if not root:
        root = "/home/qixinx/continual-learning-bench"
    root_path = Path(root).resolve()
    src_path = root_path / "src"
    for p in (str(root_path), str(src_path)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root_path


def _load_baselines(args) -> dict[str, float]:
    global _BASELINE_CACHE
    if _BASELINE_CACHE is None:
        path = os.environ.get("CODEBASE_BASELINE_ARTIFACT") or getattr(args, "codebase_baseline_artifact", None)
        _BASELINE_CACHE = load_baseline_rewards(path)
        if not _BASELINE_CACHE:
            print("[codebase_rollout] No baseline rewards loaded; gain == reward for WRITE rewards.", flush=True)
    return _BASELINE_CACHE


def _int_env_or_arg(env_name: str, args: Any, arg_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw not in (None, ""):
        return int(raw)
    return int(getattr(args, arg_name, default))


def _encode_prompt(state, messages, enable_thinking=None):
    tokenizer = state.tokenizer
    if enable_thinking is None:
        enable_thinking = bool(getattr(state.args, "arc_enable_thinking", False))
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
    return [int(t) for t in enc["input_ids"]]


def _encode_prompt_with_fifo(
    state,
    messages: list[dict[str, str]],
    *,
    max_context_tokens: int,
    reserve_tokens: int,
) -> tuple[list[int], int]:
    """Encode a full-history chat, dropping oldest messages to fit the ICL budget."""
    available = max(1, int(max_context_tokens) - int(reserve_tokens))
    truncated = 0
    while True:
        prompt_ids = _encode_prompt(state, messages)
        if len(prompt_ids) <= available or not messages:
            return prompt_ids, truncated
        messages.pop(0)
        truncated += 1


async def _infer(url: str, input_ids: list[int], sampling_params: dict):
    payload = {"input_ids": input_ids, "sampling_params": sampling_params, "return_logprob": True}
    out = await post(url, payload)
    meta = out["meta_info"]
    if meta.get("output_token_logprobs"):
        resp_ids = [item[1] for item in meta["output_token_logprobs"]]
        resp_logprobs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        resp_ids, resp_logprobs = [], []
    finish = meta.get("finish_reason", {}).get("type", "stop")
    return _strip_lone_surrogates(out.get("text", "")), resp_ids, resp_logprobs, finish


def _strip_special(text: str) -> str:
    return (text or "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


# ── 纯文本输出格式(用户决策 2026-07-13, 替换 JSON 结构化输出)─────────────────
# 动机: JSON 输出格式疑似不适合 qwen3.5 的训练。改为 mini-swe-agent 式纯文本:
# 先推理、后输出唯一一个 ```bash 代码块。仅覆盖训练路径(在 rollout 进程内 patch
# clbench task 模块的 _SYSTEM_TEMPLATE), clbench 仓库文件与离线评测不受影响。
_TEXT_ACT_SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact with a computer.

First reason about the current state and what should be done next, then output your command. \
Always reason first; after your reasoning, give exactly ONE bash code block containing ONE command \
(or commands connected with && or ||). The code block must contain only the raw command — \
no explanation inside it.

Never output a bash code block by itself: every response must start with your reasoning \
(at least one sentence analyzing the last result and explaining why you chose this command) \
before the code block.

Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```bash
your_command_here
```
</format_example>\
"""

_BASH_BLOCK_RE = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)

# 围栏代码块的语言标签(用于区分"写了块, 但标签不是 bash"这一类)。
_ANY_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)\s*\n", re.M)
# 空命令时附给模型的默认说明。对"围栏没闭合""压根没写代码块"是准确的; 对下面两类不准确。
_EMPTY_CMD_NOTE_GENERIC = "No complete ```bash code block was found in your previous response."
# opt-in(默认关): 空命令时按实际成因给反馈, 而不是一律说"没找到完整的 ```bash 块"。
# 默认关: 改反馈=改训练条件, 半途开启会让同一条 run 前后不一致; 新 run 再启用。
_MULTIBLOCK_FEEDBACK = os.environ.get("CODEBASE_MULTIBLOCK_FEEDBACK", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# WRITE 奖励模式在**模块导入期**校验, 而不是等采样到一半才发现。
# 教训(FLAME review 2026-08-08): 校验原先放在 generate() 里, 而那段整个位于 episode 级
# try/except(故障隔离)之内, 于是拼错模式名不会让作业起不来, 反而是每个 episode 完整跑完
# → 末尾抛异常 → 被隔离逻辑接住 → 伪造一条 env_error outcome、丢掉全部 WRITE 样本、
# 照常训练 ACT 样本, 变成"100% env_error 还在拿污染数据训练"。比它要防的问题更糟。
# 放在导入期还顺带覆盖 eval-only 的 run —— 那条路径根本不进写奖励分支。
_VALID_WRITE_REWARD_MODES = ("", "delta", "downstream", "gated_downstream", "gated_delta_window")
_WRITE_REWARD_MODE = os.environ.get("CODEBASE_WRITE_REWARD_MODE", "").strip().lower()
if _WRITE_REWARD_MODE not in _VALID_WRITE_REWARD_MODES:
    raise ValueError(
        f"CODEBASE_WRITE_REWARD_MODE={_WRITE_REWARD_MODE!r} is not a known WRITE reward mode. "
        f"Valid values: {_VALID_WRITE_REWARD_MODES!r} "
        "('' and 'delta' both select the downstream gain-improve reward)."
    )


def _empty_command_note(text: str) -> str:
    """按实际成因说明这一轮为什么没执行命令(由 CODEBASE_MULTIBLOCK_FEEDBACK 启用)。

    **本函数的措辞必须与本文件里 _response_from_text 的判定完全一致**:
    该函数的规则是 `command = blocks[0].strip() if len(blocks) == 1 else ""`,
    即恰好 1 个 ```bash 块才执行, 0 个或 ≥2 个一律置空。上一版从另一条分支
    (orchard, 那里的解析器有 CODEBASE_MULTIBLOCK_TAKE_FIRST 取第一块的分支)
    抄了个同名开关来抑制多块文案 —— 但本分支的解析器根本不看那个变量, 结果是
    开关一开就把**准确**的"你写了 N 个块"换成了假的"没找到完整的块"(FLAME
    review 2026-08-08, 净回归)。已撤除; 两条分支合并时应先对齐 _response_from_text,
    再决定文案, 而不是跨分支复制开关。test_empty_command_note.py 里有一条断言
    盯着解析器规则, 规则一改测试就会失败, 强制重新对齐。

    默认说明是 "No complete ```bash code block was found", 这句话在几类场景下
    帮不上忙甚至是假的, 模型无法自诊断, 于是发展出"终端坏了/块被吞了"等错误理论并反复重犯:

      · 写了 ≥2 个块 —— 这句话是**假的**(块又多又完整)。
      · 写成 ```python 等非 bash 标签 —— 这句话是真的, 但没告诉它"只有 ```bash 会被执行"。
      · 单个块但内容为空 —— 这句话也是**假的**(FLAME review 2026-08-08 指出)。

    实测(deltawin3 全部 93 个 rollout / 274282 轮, 其中判空 22992 轮):
    ≥2 块 61.4%, 非 bash 标签 18.4%, 空块 0.02%, 其余(围栏没闭合/无代码块)20.2%。

    只改反馈文本, 不改"多块判空"这条规则本身(动作语义不变)。
    验证: scripts/test_empty_command_note.py --full(全量回放 + 健全性断言)。
    """
    t = text or ""
    blocks = _BASH_BLOCK_RE.findall(t)
    n_bash = len(blocks)
    tail = "Reply with your reasoning first, then exactly ONE ```bash block containing one command."
    # 单个块但内容为空 —— 说"没找到完整的块"是假话(块找到了, 只是空的)。
    # 注意"空块 + 后面还有块"不会走到这里: 正则回溯会把两者并成 1 个内容为 ``` 的块,
    # 命令非空, 解析器会真的去执行 ``` —— 那是解析器层面的问题, 不在本函数职责内。
    if n_bash == 1 and not blocks[0].strip():
        return f"Your previous response contained an empty ```bash block, so no command ran. {tail}"
    if n_bash >= 2:
        return (
            f"Your previous response contained {n_bash} ```bash code blocks; "
            f"exactly ONE is required, so no command was executed. {tail}"
        )
    # 只有当**没有任何可执行的 bash 围栏**、却有别的语言围栏时才这么说。
    # 抑制条件按**原样大小写**比较, 与 _BASH_BLOCK_RE 的大小写敏感一致:
    #   · 小写 ```bash 却没解析出完整块 → 围栏没闭合(常被长度上限截断, 且此前可能已写完
    #     一个 ```python 块), 说"你的块不是 bash"是假话, 还会把模型从真正的修法带偏;
    #   · 大写 ```Bash → 执行正则永远匹配不到, 它**就是**"不是 ```bash 块"的一种,
    #     这句提示恰恰是它需要的。若这里也 lower 后比较, 就会把它一起抑制掉(回归)。
    # FLAME review 2026-08-08 两条合并。
    _tags = _ANY_FENCE_RE.findall(t)
    if n_bash == 0 and "bash" not in _tags and any(x != "" for x in _tags):
        return (
            "Your previous response contained a code block, but it was not a ```bash block; "
            f"only ```bash blocks are executed, so no command ran. {tail}"
        )
    return _EMPTY_CMD_NOTE_GENERIC


def _apply_text_format_override() -> None:
    """把 clbench 的 JSON system 模板换成纯文本模板(仅当前进程, 幂等)。"""
    import src.tasks.codebase_adaptation.task as _cb_task_mod

    if getattr(_cb_task_mod, "_MILES_TEXT_FORMAT_APPLIED", False):
        return
    _cb_task_mod._SYSTEM_TEMPLATE = _TEXT_ACT_SYSTEM_TEMPLATE
    _cb_task_mod._load_mswea_templates.cache_clear()
    _cb_task_mod._MILES_TEXT_FORMAT_APPLIED = True
    print("[codebase_rollout] ACT output format = plain text (bash code block); JSON schema disabled", flush=True)


# 模型偶尔会吐出孤立的 UTF-16 代理字符(如 \ud834, 音乐符号 U+1D1xx 的高位代理), 会让
# pydantic model_dump_json / JSON 序列化直接崩(surrogates not allowed), 掀翻整个训练步。
# 合法的 astral 字符在 Python str 里是单码点、不落在代理区(D800-DFFF), 所以此处只删真正孤立
# 的坏字符, 不会误伤正常表情/符号。在 _infer 出口统一清洗, 覆盖 ACT/WRITE 所有下游序列化。
_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _strip_lone_surrogates(text: str) -> str:
    return _LONE_SURROGATE_RE.sub("", text) if text else text


def _safe_path_part(value) -> str:
    text = str(value if value is not None else "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_trajectory(record: dict) -> None:
    """Dump a human-readable per-episode trajectory to CODEBASE_TRAJ_DIR."""
    root = os.environ.get("CODEBASE_TRAJ_DIR")
    if not root:
        return
    try:
        rollout_id = _safe_path_part(record.get("rollout_id"))
        episode_id = _safe_path_part(record.get("episode_id"))
        sample_index = _safe_path_part(record.get("index"))
        if record.get("evaluation"):
            dataset = _safe_path_part(record.get("eval_dataset_name"))
            traj_dir = os.path.join(root, "eval", dataset, f"rollout_{rollout_id}")
        else:
            traj_dir = os.path.join(root, "train", f"rollout_{rollout_id}")
        os.makedirs(traj_dir, exist_ok=True)
        path = os.path.join(traj_dir, f"ep_{sample_index}_episode_{episode_id}.json")
        with open(path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _json_candidates(text: str) -> list[str]:
    cleaned = _strip_special(text)
    candidates = [cleaned]
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            body = part.strip()
            if not body:
                continue
            if body.startswith("json"):
                body = body[4:].strip()
            candidates.append(body)
    return candidates


def _decode_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _json_string_unescape(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value.replace(r'\"', '"').replace(r"\n", "\n")


def _scan_unclosed_command(text: str) -> str:
    patterns = (
        r'"command"\s*:\s*"',
        r'\\"command\\"\s*:\s*\\"',
        r'\\"command\\"\s*:\s*"',
        r'"command"\s*:\s*\\"',
    )
    start = None
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.S)
        if match:
            start = match.end()
            break
    if start is None:
        return ""
    raw = text[start:].strip()
    if not raw:
        return ""
    raw = re.sub(r"\s*```.*$", "", raw, flags=re.S).strip()
    raw = re.sub(r"[}\s]+$", "", raw).strip()
    return _json_string_unescape(raw)


def _scan_string_fields(text: str, field_names: set[str]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for field in field_names:
        # Handles both normal JSON keys and common escaped-key failures such as
        # {\"thought\": ..., \"command\": ...}. It only accepts quoted string
        # values; non-string coercion is handled later by the Pydantic schema.
        patterns = (
            rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"',
            rf'\\"{re.escape(field)}\\"\s*:\s*\\"((?:\\.|[^"\\])*)\\"',
            rf'\\"{re.escape(field)}\\"\s*:\s*"((?:\\.|[^"\\])*)"',
            rf'"{re.escape(field)}"\s*:\s*\\"((?:\\.|[^"\\])*)\\"',
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.S)
            if match:
                payload[field] = _json_string_unescape(match.group(1))
                break
    if "command" in field_names and not str(payload.get("command", "")).strip():
        command = _scan_unclosed_command(text)
        if command:
            payload["command"] = command
    return payload


def _parse_json_object(text: str, field_names: set[str] | None = None) -> dict[str, Any]:
    fields = field_names or set()
    last_obj: dict[str, Any] = {}
    for candidate in _json_candidates(text):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        value = _decode_first_json_object(candidate)
        if value:
            last_obj = value
            if not fields or any(name in value for name in fields):
                return value

    if fields:
        scanned = _scan_string_fields(_strip_special(text), fields)
        if scanned:
            return scanned
    return last_obj


def _response_from_text(query, text: str):
    """纯文本解析(用户决策 2026-07-13, 取代 JSON 解析):

    响应含恰好一个 ```bash 代码块 → 块内为 command, 块外文字为 thought;
    0 个或多个块 → command 置空(task 对空命令返回 "Empty command" 提示且不消耗
    步数预算, 模型下一轮重试)。注意: 这里的解析结果只喂给 task.step 执行——
    进对话历史和训练样本的始终是模型原文(clean_act), 与解析无关。
    """
    action_cls = query.response_schema
    fields = getattr(action_cls, "model_fields", {})
    text = text or ""
    blocks = _BASH_BLOCK_RE.findall(text)
    command = blocks[0].strip() if len(blocks) == 1 else ""
    thought = _BASH_BLOCK_RE.sub("", text).strip()
    payload = {}
    if "command" in fields:
        payload["command"] = _strip_lone_surrogates(command)
    if "thought" in fields:
        payload["thought"] = _strip_lone_surrogates(thought)
    try:
        return action_cls(**payload)
    except Exception:
        return action_cls(**{field: "" for field in fields})


def _structured_act_params(sampling_params: dict[str, Any], query) -> dict[str, Any]:
    """纯文本输出模式: 不再施加 json_schema 约束解码(用户决策 2026-07-13)。"""
    return deepcopy(sampling_params)


def _pack_act_sample(state, seed, system_text: str, messages: list[dict[str, str]], trial_pos: int):
    global _MASK_GEN
    if not any(m["role"] == "assistant" for m in messages):
        return None
    if _MASK_GEN is None:
        _MASK_GEN = _QwenMaskGen(state.tokenizer, tokenizer_type="qwen3")
    prefix = [{"role": "system", "content": system_text}] if system_text else []
    ids, mask = _MASK_GEN.get_loss_mask(prefix + messages)
    response_length = _MASK_GEN.get_response_lengths([mask])[0]
    if response_length <= 0:
        return None
    return Sample(
        group_index=seed.group_index,
        index=seed.index,
        prompt="(packed codebase issue chat)",
        tokens=ids,
        response="(packed codebase issue)",
        response_length=response_length,
        loss_mask=mask[-response_length:],
        rollout_log_probs=None,
        status=Sample.Status.COMPLETED,
        metadata={"phase": "act", "trial_pos": int(trial_pos)},
    )


def _pack_write_sample(seed, prompt_ids, response_text, response_ids, response_logprobs, finish, rewrite_idx: int):
    if not response_ids:
        return None
    status = {
        "length": Sample.Status.TRUNCATED,
        "abort": Sample.Status.ABORTED,
    }.get(finish, Sample.Status.COMPLETED)
    return Sample(
        group_index=seed.group_index,
        index=seed.index,
        prompt="(codebase memory update)",
        tokens=list(prompt_ids) + list(response_ids),
        response=response_text,
        response_length=len(response_ids),
        loss_mask=[1] * len(response_ids),
        rollout_log_probs=response_logprobs,
        status=status,
        metadata={"phase": "write", "rewrite_idx": int(rewrite_idx)},
    )


# Snapshot of CLBENCH_SINGULARITY_EXEC_ARGS as it was BEFORE any task touched it
# (captured at import, prior to the first SweBenchCLTask.__init__ setdefault). The
# eval branch restores this so it never clobbers a user-supplied explicit override.
_ORIG_SINGULARITY_EXEC_ARGS = os.environ.get("CLBENCH_SINGULARITY_EXEC_ARGS")


def _make_task(args, split: str, instance_ids: list[str], stage_labels: list[str]):
    root = _clbench_root(args)
    _apply_text_format_override()
    max_steps = _int_env_or_arg("CODEBASE_MAX_STEPS_PER_ISSUE", args, "codebase_max_steps_per_issue", 40)
    seed = int(getattr(args, "codebase_seed", 42))
    if split == "train":
        # 训练走 swe_bench_cl(继承 codebase, 但 django/sympy 用官方判分, 不能用 codebase 的 pytest
        # returncode 误判)。dataset 含全部 232 池 id 即可; reset 用 _schedule_instance_ids 精确取本
        # episode 的 19 题, num_instances 在该路径下不生效, schedule=None 跳过内建排布。
        # CODEBASE_TRAIN_TASK: 训练任务族切换(2026-07-20, SWE-smith 数据扩充)。
        # swe_smith 继承 SweBenchCLTask, 判分/exec-args 行为同构, 仅数据来源不同。
        _train_task = os.environ.get("CODEBASE_TRAIN_TASK", "swe_bench_cl")
        if _train_task == "swe_smith":
            from src.tasks.swe_smith.task import SweSmithTask as _TrainTaskCls
            _default_rel = "data/swe_smith/pilot.jsonl"
        else:
            from src.tasks.swe_bench_cl.task import SweBenchCLTask as _TrainTaskCls
            _default_rel = "data/swe_bench_cl/full.jsonl"

        rel = os.environ.get("CODEBASE_TRAIN_DATASET") or getattr(
            args, "codebase_train_dataset", _default_rel)
        dataset_path = str(root / rel)
        task = _TrainTaskCls(
            dataset_path=dataset_path, schedule=None,
            max_steps_per_issue=max_steps, seed=seed,
        )
    elif split == "rebench":
        # SWE-rebench V2 eval(单题 no_memory,与 codebase heldout 同为 eval 路径)。
        # SweRebenchTask 继承 codebase generic-PR runtime,额外跑 install_config + 严格判分。
        if _ORIG_SINGULARITY_EXEC_ARGS is None:
            os.environ.pop("CLBENCH_SINGULARITY_EXEC_ARGS", None)
        else:
            os.environ["CLBENCH_SINGULARITY_EXEC_ARGS"] = _ORIG_SINGULARITY_EXEC_ARGS
        from src.tasks.swe_rebench.task import SweRebenchTask

        rel = getattr(args, "codebase_rebench_dataset", "data/swe_rebench/pilot.jsonl")
        dataset_path = str(root / rel)
        task = SweRebenchTask(
            dataset_path=dataset_path, schedule=None,
            max_steps_per_issue=max_steps, seed=seed,
        )
    else:
        # swe_bench_cl (train) writes CLBENCH_SINGULARITY_EXEC_ARGS process-globally
        # (testbed-first PATH, LANG, and notably NO --fakeroot for the SWE-bench
        # images). default_singularity_exec_args() reads that env first (and it is
        # consumed by the interactive env + the grading primitive singularity_exec —
        # NOT by singularity_start_container, which only builds the sandbox). Without
        # this reset, codebase_adaptation's eval containers would silently run under
        # swe_bench_cl's args instead of codebase's own tested defaults (--fakeroot +
        # each image's baked PATH). For today's tablib/tenacity eval images python
        # still resolves (their python is on the leaked PATH), so this is config drift
        # rather than an immediate break — but it changes codebase's tested behavior
        # and is unsafe for future images. Restore the pre-train snapshot: None ->
        # backend default; an explicit user override is preserved (not clobbered).
        # Safe because miles runs train/eval rollouts sequentially (train.py:70).
        # CODEBASE_EVAL_TASK=swe_smith(2026-07-20, pass@k eval-only 采集):
        # eval 分支跑 SWE-smith 时沿用 swe_bench_cl 系 exec args(testbed PATH,
        # 无 --fakeroot; swesmith 镜像与官方 SWE-bench 同布局), 不做 codebase 重置。
        _eval_task = os.environ.get("CODEBASE_EVAL_TASK", "codebase_adaptation")
        if _eval_task == "swe_smith":
            from src.tasks.swe_smith.task import SweSmithTask

            rel = os.environ.get("CODEBASE_EVAL_DATASET", "data/swe_smith/pilot.jsonl")
            dataset_path = str(root / rel)
            task = SweSmithTask(
                dataset_path=dataset_path, schedule=None,
                max_steps_per_issue=max_steps, seed=seed,
            )
            task.dataset_path = dataset_path
            order = type("Order", (), {"instance_ids": instance_ids, "stage_labels": stage_labels})()
            task._schedule_instance_ids = list(instance_ids)
            task._schedule_stage_lookup = stage_lookup_for(order)
            task._schedule_stage_sizes = []
            _last = None
            for label in stage_labels:
                if label != _last:
                    task._schedule_stage_sizes.append(1)
                    _last = label
                else:
                    task._schedule_stage_sizes[-1] += 1
            return task
        if _ORIG_SINGULARITY_EXEC_ARGS is None:
            os.environ.pop("CLBENCH_SINGULARITY_EXEC_ARGS", None)
        else:
            os.environ["CLBENCH_SINGULARITY_EXEC_ARGS"] = _ORIG_SINGULARITY_EXEC_ARGS
        from src.tasks.codebase_adaptation.task import CodebaseAdaptationTask

        dataset_path = str(root / "data" / "codebase_adaptation" / "final-dataset.jsonl")
        task = CodebaseAdaptationTask(
            dataset_path=dataset_path, schedule="default",
            max_steps_per_issue=max_steps, seed=seed,
        )
    task.dataset_path = dataset_path
    order = type("Order", (), {"instance_ids": instance_ids, "stage_labels": stage_labels})()
    task._schedule_instance_ids = list(instance_ids)
    task._schedule_stage_lookup = stage_lookup_for(order)
    task._schedule_stage_sizes = []
    last = None
    for label in stage_labels:
        if label != last:
            task._schedule_stage_sizes.append(1)
            last = label
        else:
            task._schedule_stage_sizes[-1] += 1
    return task


def _force_complete_current_issue(task, *, max_steps_per_issue: int):
    """End the active benchmark issue if generation attempts exhaust the task budget.

    This is a defensive guard for malformed/empty actions: the benchmark step
    budget is authoritative, and each episode must advance to the next issue
    instead of re-entering the same issue as a new ACT.
    """
    from src.interface import Observation, TaskStepResult

    instance = task.instances[task.current_issue_idx]
    step = int(getattr(task, "current_steps", 0) or 0)
    issue_record = {
        "issue_id": instance.instance_id,
        "issue_index": task.current_issue_idx,
        "canonical_issue_index": task.canonical_instance_index(task.current_issue_idx),
        "steps": max(1, step),
        "success": False,
        "tests_passed": 0,
        "tests_failed": 0,
        "timed_out": True,
        "eval_status": "rollout_step_budget",
    }
    task.issue_history.append(issue_record)
    task.interaction_trace.append(
        {
            "interaction": int(getattr(task, "total_interactions", 0) or 0),
            "issue_id": instance.instance_id,
            "issue_index": task.current_issue_idx,
            "step": step,
            "rollout_step_budget": int(max_steps_per_issue),
        }
    )
    instance_outcome = task._build_issue_instance_outcome(issue_record)

    obs = f"Rollout exhausted the issue step budget ({max_steps_per_issue}). Moving to next issue."
    task.current_issue_idx += 1
    task.current_steps = 0

    if task.current_issue_idx >= len(task.instances):
        return TaskStepResult(
            observation=Observation(content=obs + "\n\nAll issues completed!", instance_complete=True),
            next_query=None,
            done=True,
            instance_outcome=instance_outcome,
        )

    task._new_issue_pending = True
    task._update_rollout_context_for_issue(task.current_issue_idx, announce=True)
    return TaskStepResult(
        observation=Observation(content=obs, instance_complete=True),
        next_query=task._next_query(),
        done=False,
        instance_outcome=instance_outcome,
    )


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args, state, seed = input.args, input.state, input.sample
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sampling_params = deepcopy(input.sampling_params)
    write_params = deepcopy(input.sampling_params)
    write_params["max_new_tokens"] = int(getattr(args, "codebase_memory_max_tokens", 768))

    meta = seed.metadata or {}
    no_memory = os.environ.get("CODEBASE_NO_MEMORY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or bool(meta.get("no_memory"))
    if no_memory and not input.evaluation:
        raise ValueError("CODEBASE_NO_MEMORY currently supports eval-only rollouts")
    rollout_id = meta.get("_eval_rollout_id", getattr(args, "alchemy_current_rollout_id", None))
    eval_dataset_name = meta.get("_eval_dataset_name")
    split = meta.get("split") or getattr(args, "codebase_split", "train")
    order_rank = meta.get("order_rank")
    if order_rank is not None:
        shuffle_seed = int(order_rank)
    elif meta.get("shuffle_seed") is not None:
        shuffle_seed = int(meta["shuffle_seed"])
    else:
        shuffle_seed = int(getattr(args, "codebase_shuffle_seed_offset", 0)) + int(seed.group_index)
    # 训练 episode(swe_bench_cl)在 seed metadata 里直接带 instance_ids/stage_labels(预生成的
    # 9+10 episode 池, 见 scripts/swecl/gen_train_episodes.py); heldout 仍走硬编码 codebase 排布。
    meta_ids = meta.get("instance_ids")
    meta_labels = meta.get("stage_labels")
    if meta_ids and meta_labels:
        order = EpisodeOrder(instance_ids=list(meta_ids), stage_labels=list(meta_labels))
    else:
        order = build_episode_order(
            split=split,
            shuffle_seed=shuffle_seed,
            order_rank=None if order_rank is None else int(order_rank),
        )
    baselines = _load_baselines(args)

    task = _make_task(args, split, order.instance_ids, order.stage_labels)
    query = await asyncio.to_thread(task.reset)
    feedback_text = None
    memory = ""
    episode_messages: list[dict[str, str]] = []
    context_truncation_count = 0
    max_context_tokens = _int_env_or_arg(
        "CODEBASE_CONTEXT_MAX_TOKENS",
        args,
        "codebase_context_max_tokens",
        240000,
    )
    context_reserve_tokens = _int_env_or_arg(
        "CODEBASE_CONTEXT_RESERVE_TOKENS",
        args,
        "codebase_context_reserve_tokens",
        500,
    )
    samples: list[Sample] = []
    write_points: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    summaries: list[str] = []
    write_audit: list[dict[str, Any]] = []
    episode_id = f"{split}:{shuffle_seed}:{seed.group_index}"
    max_steps_per_issue = _int_env_or_arg("CODEBASE_MAX_STEPS_PER_ISSUE", args, "codebase_max_steps_per_issue", 40)
    instance_cap = _int_env_or_arg("CODEBASE_NUM_ACTS_CAP", args, "codebase_num_acts_cap", 0)

    try:
        while query is not None:
            if instance_cap and len(outcomes) >= instance_cap:
                break
            trial_pos = int((query.metadata or {}).get("active_issue_index", len(outcomes)))
            instance_id = str(query.instance_id)
            messages = episode_messages if no_memory else []
            trial_messages: list[dict[str, str]] = []
            transcript: list[dict[str, Any]] = []
            memory_in = memory
            done = False
            step_result = None
            first_turn = True
            step_budget_exhausted = False

            for turn_idx in range(max_steps_per_issue):
                if feedback_text:
                    feedback_message = {
                        "role": "user",
                        "content": f"FEEDBACK: {feedback_text.strip()}",
                    }
                    messages.append(feedback_message)
                    trial_messages.append(dict(feedback_message))
                if no_memory:
                    user_content = query.prompt or "(no content)"
                else:
                    user_content = build_act_user_content(
                        memory=memory,
                        query_prompt=query.prompt,
                        include_memory=first_turn,
                    )
                # 每轮向 agent 汇报剩余预算(用户要求, alchemy takeaway)。
                # 口径 = 真实轮次(用户裁决 2026-07-13): 每一轮生成都消耗预算, 坏命令也算——
                # 这才是真正卡住 agent 的约束(此前按 task 步数报, 坏命令不减, 会虚报余量)。
                turns_left = max_steps_per_issue - turn_idx
                user_content = (
                    f"{user_content}\n\n"
                    f"[Budget] You have {turns_left} of {max_steps_per_issue} turns remaining for this issue."
                )
                messages.append({"role": "user", "content": user_content})
                trial_messages.append({"role": "user", "content": user_content})

                truncated_now = 0
                if no_memory:
                    prompt_ids, truncated_now = _encode_prompt_with_fifo(
                        state,
                        messages,
                        max_context_tokens=max_context_tokens,
                        reserve_tokens=context_reserve_tokens,
                    )
                    context_truncation_count += truncated_now
                else:
                    prompt_ids = _encode_prompt(state, messages)
                act_text, _resp_ids, _resp_lps, act_finish = await _infer(
                    url,
                    prompt_ids,
                    _structured_act_params(sampling_params, query),
                )
                clean_act = _strip_special(act_text)
                action = _response_from_text(query, clean_act)
                response = _clbench_response(action, {"raw_response": clean_act, "finish": act_finish})

                # 训练崩塌修复: assistant 上下文/训练文本必须是模型真实采样的原文(clean_act),
                # 不能用 model_dump_json() 的规范化 JSON——parse 失败时那会变成 fallback 字面量
                # '{"thought": "No parseable...", "command": ""}', 被 _pack_act_sample 当作策略输出
                # 训练, 一次更新就教会模型输出空命令(实测 step1 起 completed 0/160, ACT advantage 全 0)。
                # 离线 clbench 无此问题: 它走 provider 强制 structured output, parse 不会失败且无训练。
                # 结构化 action 仍保留在 transcript 的 "action" 字段(_jsonable)供审计。
                assistant_context = clean_act
                messages.append({"role": "assistant", "content": assistant_context})
                trial_messages.append({"role": "assistant", "content": assistant_context})
                first_turn = False

                step_result = await asyncio.to_thread(task.step, response)
                obs = step_result.observation
                transcript.append(
                    {
                        "step": len(transcript),
                        "trial_pos": trial_pos,
                        "instance_id": instance_id,
                        "memory_in": memory_in if len(transcript) == 0 else None,
                        "user": user_content,
                        "assistant": clean_act,
                        "assistant_context": assistant_context if no_memory else None,
                        "action": _jsonable(action),
                        "act_finish": act_finish,
                        "context_tokens": len(prompt_ids) if no_memory else None,
                        "truncated_messages": truncated_now if no_memory else 0,
                        "observation": obs.content,
                        "observation_metadata": _jsonable(obs.metadata),
                        "instance_complete": bool(obs.instance_complete),
                    }
                )
                if step_result.done or obs.instance_complete:
                    done = step_result.done
                    feedback_text = obs.content if no_memory else None
                    break
                if step_result.next_query is None:
                    done = True
                    feedback_text = obs.content if no_memory else None
                    break
                feedback_text = obs.content
                # 命令为空时补一句客观事实(没找到完整 bash 块); 仅当生成确实被长度上限
                # 掐断(finish=length)时, 才提"可能是超长"——超长只是可能原因之一, 不断言。
                if not str(getattr(action, "command", "") or "").strip():
                    _note = (
                        _empty_command_note(clean_act)
                        if _MULTIBLOCK_FEEDBACK
                        else _EMPTY_CMD_NOTE_GENERIC
                    )
                    if act_finish == "length":
                        _note += (
                            f" It may have been cut off by the "
                            f"{sampling_params.get('max_new_tokens', 'response')}-token response limit."
                        )
                    feedback_text = f"{obs.content}\n{_note}"
                query = step_result.next_query
            else:
                step_budget_exhausted = True

            if (
                step_budget_exhausted
                and step_result is not None
                and not step_result.done
                and not step_result.observation.instance_complete
            ):
                step_result = _force_complete_current_issue(task, max_steps_per_issue=max_steps_per_issue)
                done = step_result.done
                feedback_text = step_result.observation.content if no_memory else None

            act_sample = _pack_act_sample(
                state,
                seed,
                "",
                trial_messages,
                trial_pos,
            )
            # Bound over-length ACT samples with miles' native truncation: tail-trim
            # the response to fit seq_length, or drop the whole sample if the prompt
            # alone already exceeds it. Prevents a single >seq_length sample from
            # landing in an oversized micro-batch and OOM-ing the train step.
            # 用户要求(2026-07-13): eval 不受任何训练侧长度限制 —— evaluation 时
            # 截断/丢弃/超长打标一概不执行(eval 样本只用于算分, 不进训练批)。
            if act_sample is not None and not input.evaluation:
                # wandb 超长比例统计用(用户指令 2026-07-13): 截断/丢弃前打标
                act_sample.metadata = {
                    **(act_sample.metadata or {}),
                    "overlong_pre_truncate": len(act_sample.tokens) > int(args.seq_length),
                }
                _kept = truncate_samples_by_total_tokens(
                    [act_sample], int(args.seq_length), state.tokenizer
                )
                act_sample = _kept[0] if _kept else None
            if step_result is not None and step_result.instance_outcome is not None:
                outcome_obj = step_result.instance_outcome
                reward = float(outcome_obj.reward)
                # train reward 与真实轮次对齐(用户裁决 2026-07-13): 训练时 regret 用轮数
                # (坏命令也算一轮)镜像官方公式 solved: regret=t-1, reward=1-regret/40;
                # 失败仍为 0。eval(input.evaluation)保持 clbench 官方步数口径, 不动。
                if not input.evaluation and bool(getattr(outcome_obj, "success", False)):
                    _turns_used = max(1, min(len(transcript), max_steps_per_issue))
                    reward = round(1.0 - (_turns_used - 1) / max_steps_per_issue, 4)
                baseline = float(baselines.get(outcome_obj.instance_id, 0.0))
                outcome = {
                    "instance_id": outcome_obj.instance_id,
                    "reward": reward,
                    "baseline_reward": baseline,
                    "gain": reward - baseline,
                    "success": outcome_obj.success,
                    "steps": (outcome_obj.metadata or {}).get("steps"),
                    "turns": len(transcript),
                    "eval_status": (outcome_obj.metadata or {}).get("eval_status"),
                    "trial_pos": trial_pos,
                }
            else:
                reward = 0.0
                outcome = {
                    "instance_id": instance_id,
                    "reward": 0.0,
                    "baseline_reward": baselines.get(instance_id, 0.0),
                    "gain": -baselines.get(instance_id, 0.0),
                    "turns": len(transcript),
                    "trial_pos": trial_pos,
                }

            if act_sample is not None:
                act_sample.reward = reward
                act_sample.metadata = {
                    **meta,
                    **(act_sample.metadata or {}),
                    "episode_id": episode_id,
                    "no_memory": no_memory,
                    **outcome,
                }
                samples.append(act_sample)
            outcomes.append(outcome)

            trial_record = {
                "trial_pos": trial_pos,
                "instance_id": instance_id,
                "stage": order.stage_labels[trial_pos] if trial_pos < len(order.stage_labels) else None,
                "memory_in": memory_in,
                "turns": transcript,
                "outcome": outcome,
            }

            if no_memory:
                trial_record["write"] = None
                trials.append(trial_record)
                if done or step_result is None or step_result.next_query is None:
                    break
                query = step_result.next_query
                continue

            # 判分结果必须让 WRITE(memory 重写)看到: 判分在模型最后一个动作之后才产生,
            # 不追加的话 memory 写手不知道这道题的做法到底成没成。只进 WRITE 输入,
            # 不进 ACT 训练样本(_pack_act_sample 在上面已完成, 训练 token 不含这条),
            # 也不进下一题开头(replace 模式下一题看不到上一题内容, 单独给结论无意义)。
            final_feedback = ""
            if step_result is not None and getattr(step_result, "observation", None) is not None:
                final_feedback = (step_result.observation.content or "").strip()
            if final_feedback:
                trial_messages.append(
                    {"role": "user", "content": f"FEEDBACK: {final_feedback}"}
                )
            # WRITE prompt v3 / WRITE 专属 thinking(2026-07-19, 离线 A/B 验证后落地):
            # 均为 env 开关且默认关, 现役 run 重启口径不漂移。
            _write_v3 = os.environ.get("CODEBASE_WRITE_PROMPT_V3", "").strip().lower() in {"1", "true", "yes", "on"}
            _write_think = os.environ.get("CODEBASE_WRITE_THINKING", "").strip().lower() in {"1", "true", "yes", "on"}
            write_messages = build_write_messages(
                previous_memory=memory,
                instance_messages=trial_messages,
                v3=_write_v3,
            )
            write_ids = _encode_prompt(state, write_messages, enable_thinking=True if _write_think else None)
            previous_memory = memory
            write_text, write_resp_ids, write_lps, write_finish = await _infer(url, write_ids, write_params)
            write_sample = _pack_write_sample(seed, write_ids, write_text, write_resp_ids, write_lps, write_finish, trial_pos)
            # WRITE 样本超长处理(用户指令 2026-07-13, 与 ACT 同款 miles 原生语义):
            # prompt 单独超 seq_length -> 整条丢弃; 总长超 -> 尾裁; reward 不动。
            # 丢弃只影响训练样本; memory 本身照常更新(改写发生在推理侧, 与训练无关)。
            # eval 时同样跳过(用户要求: eval 不受训练侧长度限制)。
            train_write_sample = None
            if write_sample is not None and not input.evaluation:
                write_sample.metadata = {
                    **(write_sample.metadata or {}),
                    "overlong_pre_truncate": len(write_sample.tokens) > int(args.seq_length),
                }
                _kept_w = truncate_samples_by_total_tokens(
                    [write_sample], int(args.seq_length), state.tokenizer
                )
                train_write_sample = _kept_w[0] if _kept_w else None
            next_memory = previous_memory
            if write_sample is not None:
                # thinking 模式下 <think>...</think> 只训不外泄: 进记忆/下一题上下文的
                # 一律是 </think> 之后的可见正文; think 块未闭合(截断)视为无有效输出。
                _visible = write_text
                if "</think>" in _visible:
                    _visible = _visible.split("</think>", 1)[1]
                elif "<think>" in _visible:
                    _visible = ""
                next_memory = _strip_special(_visible) or previous_memory
            write_record = {
                "rewrite_idx": trial_pos,
                "previous_memory": previous_memory,
                "input_messages": write_messages,
                "raw_output": write_text,
                "memory": next_memory,
                "finish": write_finish,
                "response_tokens": len(write_resp_ids),
                "trained": False,
                "write_reward": None,
                "write_signal": "downstream_gain_improve",
            }
            trial_record["write"] = write_record
            write_audit.append(write_record)
            if write_sample is not None:
                memory = next_memory
                summaries.append(memory)
                if not input.evaluation and train_write_sample is not None:
                    write_points.append({"sample": train_write_sample, "rewrite_idx": trial_pos, "audit": write_record})
            else:
                summaries.append(memory)
            trials.append(trial_record)

            if done or step_result is None or step_result.next_query is None:
                break
            feedback_text = None
            query = step_result.next_query

        if not input.evaluation:
            gains = [float(o.get("gain", 0.0)) for o in outcomes]
            # ACT-only 实验开关(2026-07-14 用户决策): WRITE 照常发生(记忆继续写/用),
            # 但 WRITE 样本不进训练集、不给 reward——只训 ACT, 观察记忆行为的自然演化。
            _act_only = os.environ.get("CODEBASE_TRAIN_ACT_ONLY", "").strip().lower() in {
                "1", "true", "yes", "on",
            }
            if _act_only:
                for wp in write_points:
                    wp["audit"]["trained"] = False
                    wp["audit"]["act_only_skip"] = True
                write_points = []
            # WRITE reward 模式(三组消融, 与 alchemy 对齐; CODEBASE_WRITE_REWARD_MODE):
            #   delta(缺省/空)   R(M_k)=mean(gain[k+1..])-mean(gain[..k])  —— 均值回归 bias 对照
            #   downstream        R(M_k)=reward[k+1]                        —— 裸下游, 无 format 无 delta
            #   gated_downstream  R(M_k)=format_ok*(reward[k+1]+bonus)      —— 严格 format 门控(2026-07-17)
            # 用导入期已校验过的那份, 不再重读环境变量 —— 两处各读一次会让"校验过的值"
            # 和"实际分派用的值"有机会分叉。
            _write_mode = _WRITE_REWARD_MODE
            rewards_seq = [float(o.get("reward", 0.0)) for o in outcomes]
            if _write_mode == "gated_downstream":
                from examples.codebase_adaption.codebase_advantage import (
                    gated_downstream_rewards,
                    memory_format_ok,
                )
                fmt_by_k = {
                    int(wp["rewrite_idx"]): memory_format_ok(
                        wp["audit"].get("memory") or "",
                        wp["audit"].get("previous_memory") or "",
                    )
                    for wp in write_points
                }
                write_rewards = gated_downstream_rewards(
                    rewards_seq,
                    fmt_by_k,
                    bonus=float(os.environ.get("CODEBASE_WRITE_FORMAT_BONUS", "0.1")),
                )
                _write_signal = "format_gated_downstream"
            elif _write_mode == "gated_delta_window":
                # R(M_k)=mean(reward[k+1..k+K])-mean(reward[k-K+1..k])+bonus*format_ok
                # 加法门控: 窗口差分可为负, 乘法门控会让"写坏格式"(得0)优于
                # "写对格式但后续下滑"(得负), 使格式失败变成躲避惩罚的手段。
                from examples.codebase_adaption.codebase_advantage import (
                    gated_delta_window_rewards,
                    memory_format_ok,
                )
                fmt_by_k = {
                    int(wp["rewrite_idx"]): memory_format_ok(
                        wp["audit"].get("memory") or "",
                        wp["audit"].get("previous_memory") or "",
                    )
                    for wp in write_points
                }
                write_rewards = gated_delta_window_rewards(
                    rewards_seq,
                    fmt_by_k,
                    window=int(os.environ.get("CODEBASE_WRITE_DELTA_WINDOW", "3")),
                    bonus=float(os.environ.get("CODEBASE_WRITE_FORMAT_BONUS", "0.1")),
                )
                _write_signal = "gated_delta_window"
            elif _write_mode == "downstream":
                # 裸下游: R(M_k)=reward[k+1]; 末位无下游则省略。GRPO 组内归一化承担难度。
                write_rewards = {
                    k: rewards_seq[k + 1] for k in range(len(rewards_seq) - 1)
                }
                fmt_by_k = None
                _write_signal = "downstream_raw"
            elif _write_mode in ("", "delta"):
                fmt_by_k = None
                write_rewards = downstream_improve_rewards(
                    gains,
                    window=int(getattr(args, "codebase_write_improve_k", 1)),
                    k0_mode=str(getattr(args, "codebase_write_k0_mode", "improve")),
                )
                _write_signal = "downstream_gain_improve"
            else:
                # 走不到: 模块导入期已校验过 _WRITE_REWARD_MODE(见文件顶部)。留着是为了
                # 万一将来加了新模式却忘了在这里分派, 能立刻炸而不是静默落到 delta。
                # 注意本段位于 episode 级 try/except 之内, 抛在这里只会被故障隔离接住,
                # 所以真正的校验必须留在导入期, 不能只靠这一条。
                raise AssertionError(
                    f"WRITE reward mode {_write_mode!r} passed import-time validation but has no "
                    "dispatch branch here — add one."
                )
            for wp in write_points:
                k = int(wp["rewrite_idx"])
                if k not in write_rewards:
                    wp["audit"]["no_downstream"] = k >= len(gains) - 1
                    # 跳过项也要刷新标签: :1136 的初值写死成 downstream_gain_improve,
                    # 不刷新会让未训练条目在轨迹里谎报成另一种奖励模式(deltawin3 的
                    # rollout_124 里 192 条 audit 有 57 条如此)。同 :1239 那条注释的坑。
                    wp["audit"]["write_signal"] = _write_signal
                    continue
                sample = wp["sample"]
                sample.reward = float(write_rewards[k])
                wp["audit"]["trained"] = True
                wp["audit"]["write_reward"] = float(write_rewards[k])
                wp["audit"]["downstream_trial_pos"] = k + 1
                # 审计标签必须无条件刷新: :856 的初值是 delta 名字, 只在 gated 分支
                # 刷新曾导致 downstream 组轨迹标签撒谎(2026-07-18 误报事故)
                wp["audit"]["write_signal"] = _write_signal
                if fmt_by_k is not None:
                    wp["audit"]["format_ok"] = bool(fmt_by_k.get(k))
                sample.metadata = {
                    **meta,
                    **(sample.metadata or {}),
                    "episode_id": episode_id,
                    "downstream_trial_pos": k + 1,
                    "write_signal": _write_signal,
                }
                samples.append(sample)
    except Exception as episode_exc:
        # 故障隔离(2026-07-15): 单 episode 的环境/容器故障(如 apptainer sandbox 构建
        # 偶发 FATAL)不允许炸掉整个训练 job。当前题记 env_error(0 分), 保留本 episode
        # 已完成各题的样本, 提前结束该 episode。真正的代码 bug 也会走到这里——所以必须
        # 打全量 traceback, 且 metrics 里能看到 env_error 率(见 outcome.eval_status)。
        import traceback as _tb
        _loc = locals()
        _iid = str(_loc.get("instance_id", "?"))
        print(
            f"[codebase_rollout] EPISODE ABORTED by exception (episode_id={episode_id}, "
            f"instance={_iid}): {episode_exc}\n" + _tb.format_exc(),
            flush=True,
        )
        try:
            outcomes.append({
                "instance_id": _iid,
                "reward": 0.0,
                "baseline_reward": baselines.get(_iid, 0.0),
                "gain": -baselines.get(_iid, 0.0),
                "success": False,
                "turns": len(_loc.get("transcript") or []),
                "trial_pos": int(_loc.get("trial_pos", len(outcomes))),
                "eval_status": "env_error",
            })
        except Exception:
            pass
    finally:
        try:
            task._cleanup_container()
        except Exception:
            pass

    for sample in samples:
        sample.metadata = {
            **(sample.metadata or {}),
            "codebase_episode": {
                "split": split,
                "shuffle_seed": shuffle_seed,
                "order_rank": order_rank,
                "instance_ids": order.instance_ids,
                "outcomes": outcomes,
            },
        }
    _write_trajectory(
        {
            "episode_id": episode_id,
            "group_index": seed.group_index,
            "index": seed.index,
            "rollout_id": rollout_id,
            "eval_dataset_name": eval_dataset_name,
            "evaluation": bool(input.evaluation),
            "no_memory": no_memory,
            "split": split,
            "shuffle_seed": shuffle_seed,
            "order_rank": order_rank,
            "instance_ids": order.instance_ids,
            "stage_labels": order.stage_labels,
            "max_steps_per_issue": max_steps_per_issue,
            "instance_cap": instance_cap,
            "context_max_tokens": max_context_tokens if no_memory else None,
            "context_reserve_tokens": context_reserve_tokens if no_memory else None,
            "context_truncation_count": context_truncation_count,
            "outcomes": outcomes,
            "summaries": summaries,
            "write_audit": write_audit,
            "trials": trials,
        }
    )
    return GenerateFnOutput(samples=samples)


def _clbench_response(action, metadata):
    from src.interface import Response

    return Response(action=action, metadata=metadata)
