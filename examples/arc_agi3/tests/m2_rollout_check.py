"""M2 rollout-isolation check: run `arc_rollout.generate` once against a real ARC game and a
standalone SGLang server (127.0.0.1:30000), then inspect the produced per-turn training samples.

Validates the rollout independently of the Megatron training stack. Run INSIDE the container with
the smoke's env (PYTHONPATH, PYTHONUSERBASE, ARC_AGI3_REPO, ARC_API_KEY).
"""
import asyncio
import os
import sys
from argparse import Namespace

sys.path.insert(0, "/home/qixinx/miles")

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.types import Sample

from examples.arc_agi3 import arc_rollout

CKPT = "/data/hf_cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_TURNS = 4

args = Namespace(
    hf_checkpoint=CKPT,
    chat_template_path=None,
    sglang_server_concurrency=4,
    rollout_num_gpus=1,
    rollout_num_gpus_per_engine=1,
    rollout_temperature=0.7,
    rollout_top_p=1.0,
    rollout_top_k=-1,
    rollout_max_response_len=1024,
    arc_enable_thinking=False,
    rollout_stop=None,
    rollout_stop_token_ids=None,
    rollout_skip_special_tokens=False,
    custom_generate_function_path=None,
    sglang_router_ip="127.0.0.1",
    sglang_router_port=30000,
    max_turns=MAX_TURNS,
    partial_rollout=False,
    use_distributed_post=False,
)


def main():
    from miles.utils.http_utils import init_http_client

    init_http_client(args)  # normally done in RolloutManager.__init__
    state = GenerateState(args)
    seed = Sample(
        group_index=0,
        index=0,
        metadata={"game_id": os.environ.get("DEMO_GAME", "ar25-0c556536"), "max_actions": MAX_TURNS},
    )
    inp = GenerateFnInput(state=state, sample=seed, sampling_params=state.sampling_params, evaluation=False)

    out = asyncio.run(arc_rollout.generate(inp))
    samples = out.samples
    print(f"\n=== episode produced {len(samples)} turn-sample(s) ===")
    for i, s in enumerate(samples):
        mm = sorted((s.multimodal_train_inputs or {}).keys())
        len_ok = len(s.loss_mask) == s.response_length == len(s.rollout_log_probs)
        print(f"turn {i}: resp_len={s.response_length} loss_mask={len(s.loss_mask)} "
              f"logprobs={len(s.rollout_log_probs)} tokens={len(s.tokens)} mm={mm} "
              f"reward={s.reward} status={s.status.value} len_ok={len_ok}")
        print("   tail:", repr(s.response[-160:]))
    if samples:
        print("\nmetadata[0]:", samples[0].metadata)

    assert samples, "no samples produced"
    for s in samples:
        assert len(s.loss_mask) == s.response_length == len(s.rollout_log_probs), "length mismatch"
        assert s.multimodal_train_inputs and "pixel_values" in s.multimodal_train_inputs, "missing pixel_values"
        assert s.reward is not None, "reward not set"
    print("\nROLLOUT CHECK PASS")


if __name__ == "__main__":
    main()
