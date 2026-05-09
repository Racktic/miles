"""Compatibility helpers for Miles SWE-agent training through Brew Modal."""

from __future__ import annotations

from uda.swe_agent import generate_with_swe_agent as impl


def _ensure(args):
    if not hasattr(args, "wandb_experiment_name"):
        setattr(
            args,
            "wandb_experiment_name",
            getattr(args, "wandb_group", None)
            or getattr(args, "wandb_exp_name", None)
            or "miles_swe_agent_brew_modal",
        )

    if getattr(args, "sglang_context_length", None) is None:
        context_len = getattr(args, "rollout_max_context_len", None)
        if context_len is None:
            prompt_len = getattr(args, "rollout_max_prompt_len", None)
            response_len = getattr(args, "rollout_max_response_len", None)
            if prompt_len is not None and response_len is not None:
                context_len = prompt_len + response_len
        setattr(args, "sglang_context_length", context_len or 8192)

    if not hasattr(args, "agent_filter_overlong"):
        setattr(args, "agent_filter_overlong", False)

    if not hasattr(args, "agent_truncation_penalty"):
        setattr(args, "agent_truncation_penalty", 0.0)

    return args


async def generate(args, sample, sampling_params):
    _ensure(args)
    return await impl.generate(args, sample, sampling_params)


async def reward_func(args, sample, **kwargs):
    _ensure(args)
    return await impl.reward_func(args, sample, **kwargs)


def dynamic_filter(args, samples, **kwargs):
    _ensure(args)
    return impl.dynamic_filter(args, samples, **kwargs)


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    _ensure(args)
    return impl.generate_rollout(args, rollout_id, data_buffer, evaluation=evaluation)
