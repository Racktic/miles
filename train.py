import asyncio
import os

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.async_utils import eager_create_task
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import finish_tracking, init_tracking


async def train(args):
    configure_logger()
    eval_only = args.num_rollout == 0 and args.eval_interval is not None
    if eval_only:
        # There is no colocated training model in eval-only mode, so releasing
        # rollout memory only adds a release/resume cycle before first use.
        args.offload_rollout = False

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Eval-only runs use the rollout checkpoint directly and must not construct a
    # training optimizer. With num_rollout=0, Megatron derives zero decay steps
    # and rejects the optimizer configuration before evaluation can start.
    if eval_only:
        # Rollout servers start asynchronously. Normal training waits for them
        # while constructing the actor; eval-only must provide that barrier.
        engines_and_lock = await rollout_manager.get_updatable_engines_and_lock.remote()
        for _ in range(120):
            health = await asyncio.gather(
                *(engine.health_generate.remote(timeout=5) for engine in engines_and_lock.rollout_engines),
                return_exceptions=True,
            )
            if health and all(result is True for result in health):
                break
            await asyncio.sleep(5)
        else:
            raise TimeoutError("Rollout engines did not become HTTP-ready within 10 minutes")
        await rollout_manager.eval.remote(rollout_id=0)
        await rollout_manager.dispose.remote()
        return

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(action="compare")

    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    async def offload_train():
        if args.offload_train:
            if args.use_critic:
                await critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    await actor_model.offload()
            else:
                await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            await actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            await critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            await rollout_manager.save.remote(rollout_id)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            await rollout_manager.eval.remote(rollout_id)

        rollout_data_ref = await rollout_manager.generate.remote(rollout_id)

        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            await rollout_manager.offload.remote(tags=offload_tags)

        if args.use_critic:
            critic_task = await eager_create_task(critic_model.train(rollout_id, rollout_data_ref))
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_ref)
            await critic_task
        else:
            await actor_model.train(rollout_id, rollout_data_ref)

        save_num_rollout = None if os.environ.get("MILES_DISABLE_FINAL_SAVE") else args.num_rollout
        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, save_num_rollout):
            await save(rollout_id)

        await offload_train()
        if args.offload_rollout:
            await rollout_manager.onload_weights.remote()
        await actor_model.update_weights()
        if args.offload_rollout:
            await rollout_manager.onload_kv.remote()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await rollout_manager.eval.remote(rollout_id)

    # Secondary Ray processes own the rollout and optimizer metric streams.
    # Flush them before returning to the driver, whose finally block closes the
    # primary W&B process and marks the shared run complete.
    await rollout_manager.dispose.remote()
    await actor_model.finish_tracking()
    if critic_model is not None:
        await critic_model.finish_tracking()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
