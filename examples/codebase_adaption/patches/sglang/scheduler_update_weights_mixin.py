from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    PostProcessWeightsReqInput,
    PostProcessWeightsReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:
    # Ported from sglang PR #27140 (fix: recapture CUDA graphs for RL online
    # weight updates). Hybrid-model conv/ssm state buffers live under the
    # kv_cache memory-saver tag; after a release/resume cycle they can land at
    # new addresses while captured CUDA graphs still replay the old pointers.
    # Weight updates mark graphs stale; the resume that follows (miles:
    # onload_weights -> update_weights -> onload_kv) recaptures them once.
    cuda_graphs_need_recapture: bool = False

    def flush_cache_after_weight_update(self: Scheduler, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    def recapture_cuda_graphs_after_weight_update(self: Scheduler) -> None:
        # Recapture TP worker CUDA graphs
        tp_model_runner = self.tp_worker.model_runner
        if tp_model_runner.graph_runner is not None:
            logger.info("Recapturing TP worker CUDA graphs after weight update")
            for graph in tp_model_runner.graph_runner.graphs.values():
                graph.reset()
            tp_model_runner.graph_runner.graphs.clear()
            tp_model_runner.graph_runner.output_buffers.clear()
            tp_model_runner.graph_runner.capture()
        if getattr(tp_model_runner, "piecewise_cuda_graph_runner", None) is not None:
            logger.info(
                "Recapturing TP worker piecewise CUDA graphs after weight update"
            )
            tp_model_runner.init_piecewise_cuda_graphs()

        # Recapture draft worker CUDA graphs if present
        if self.draft_worker is not None:
            draft_model_runner = getattr(self.draft_worker, "model_runner", None)
            if draft_model_runner is not None:
                if draft_model_runner.graph_runner is not None:
                    logger.info(
                        "Recapturing draft worker CUDA graphs after weight update"
                    )
                    for graph in draft_model_runner.graph_runner.graphs.values():
                        graph.reset()
                    draft_model_runner.graph_runner.graphs.clear()
                    draft_model_runner.graph_runner.output_buffers.clear()
                    draft_model_runner.graph_runner.capture()
                if (
                    getattr(draft_model_runner, "piecewise_cuda_graph_runner", None)
                    is not None
                ):
                    logger.info(
                        "Recapturing draft worker piecewise CUDA graphs after weight update"
                    )
                    draft_model_runner.init_piecewise_cuda_graphs()

    def mark_cuda_graphs_stale(self: Scheduler) -> None:
        self.cuda_graphs_need_recapture = True

    def _quiesce_for_weight_update(self: Scheduler):
        """Drain in-flight forward work before any NCCL weight mutation.
        Synchronize forward_stream and schedule_stream to ensure all ranks are quiescent.
        """
        if self.enable_overlap:
            self.forward_stream.synchronize()
        self.schedule_stream.synchronize()
        if self.tp_cpu_group is not None:
            torch.distributed.barrier(group=self.tp_cpu_group)

    def update_weights_from_disk(
        self: Scheduler, recv_req: UpdateWeightFromDiskReqInput
    ):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        tp_success = success
        if success and self.draft_worker is not None:
            success, message = self.draft_worker.update_weights_from_disk(recv_req)
        if tp_success:
            self.flush_cache_after_weight_update(recv_req)
            self.mark_cuda_graphs_stale()
        if not success:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(
        self: Scheduler, recv_req: InitWeightsUpdateGroupReqInput
    ):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(
        self: Scheduler, recv_req: DestroyWeightsUpdateGroupReqInput
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        self._quiesce_for_weight_update()
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            self.flush_cache_after_weight_update(recv_req)
            self.mark_cuda_graphs_stale()
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(
        self: Scheduler, recv_req: UpdateWeightsFromTensorReqInput
    ):
        """Update the online model parameter from tensors."""
        self._quiesce_for_weight_update()
        if recv_req.disable_draft_model:
            worker = self.tp_worker
        else:
            worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        if success:
            self.flush_cache_after_weight_update(recv_req)
            self.mark_cuda_graphs_stale()
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(
        self: Scheduler, recv_req: UpdateWeightsFromIPCReqInput
    ):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        self._quiesce_for_weight_update()
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        tp_success = success
        if success and self.draft_worker is not None:
            success, message = self.draft_worker.update_weights_from_ipc(recv_req)
        if tp_success:
            self.flush_cache_after_weight_update(recv_req)
            self.mark_cuda_graphs_stale()
        if not success:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def post_process_weights(self, recv_req: PostProcessWeightsReqInput):
        """Optional post-processing for updated weights (e.g., Marlin conversion)."""
        self._quiesce_for_weight_update()
        success, message = self.tp_worker.post_process_weights(recv_req)
        if self.tp_cpu_group is not None:
            torch.distributed.barrier(group=self.tp_cpu_group)
        return PostProcessWeightsReqOutput(success, message)

    def get_weights_by_name(self: Scheduler, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if hasattr(self, "disagg_decode_prealloc_queue"):
                    self.disagg_decode_prealloc_queue.release_memory_occupation()
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                if hasattr(self, "disagg_prefill_bootstrap_queue"):
                    self.disagg_prefill_bootstrap_queue.release_memory_occupation()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                if hasattr(self, "disagg_decode_prealloc_queue"):
                    self.disagg_decode_prealloc_queue.resume_memory_occupation()
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                if hasattr(self, "disagg_prefill_bootstrap_queue"):
                    self.disagg_prefill_bootstrap_queue.resume_memory_occupation()

            # Flush again on RESUMED memory. The flush issued while the pool
            # was paused (e.g. at the start of a weight update) wrote its
            # resets into physical pages that were released, so bookkeeping
            # tensors under the kv_cache tag (req_to_token, mamba index
            # mappings) come back as garbage after resume. Re-running the
            # flush here rewrites them on live pages.
            logger.info("Post-resume cache flush (rewrite pool bookkeeping on live memory)")
            self.flush_cache()

            # Shotgun-zero every kv_cache-tag tensor whose contents could be
            # read before being rewritten (flush only resets bookkeeping
            # indices, never contents; the mamba pad slot 0 in particular is
            # never re-allocated so alloc-time zeroing never touches it).
            try:
                mr = self.tp_worker.model_runner
                pool = mr.req_to_token_pool
                pool.req_to_token.zero_()
                mamba_pool = getattr(pool, "mamba_pool", None)
                if mamba_pool is not None:
                    for _t in mamba_pool.mamba_cache.conv:
                        _t.zero_()
                    mamba_pool.mamba_cache.temporal.zero_()
                kv_pool = getattr(mr, "token_to_kv_pool", None)
                for _attr in ("k_buffer", "v_buffer", "kv_buffer"):
                    _bufs = getattr(kv_pool, _attr, None)
                    if _bufs is not None:
                        for _t in _bufs:
                            _t.zero_()
                logger.info("Post-resume shotgun zero of kv_cache-tag contents done")
            except Exception as exc:
                logger.warning(f"Post-resume shotgun zero failed (non-fatal): {exc}")

        if self.cuda_graphs_need_recapture:
            self.recapture_cuda_graphs_after_weight_update()
            self.cuda_graphs_need_recapture = False

        self._log_model_state_checksums(f"resume(tags={sorted(tags)})")

        return ResumeMemoryOccupationReqOutput()

    def _log_model_state_checksums(self: Scheduler, tag_str: str) -> None:
        """Diagnostic fingerprints: under lr=0 every buffer/parameter sum must
        be identical across sync cycles; any drift names the corrupted tensor."""
        try:
            model = self.tp_worker.model_runner.model
            with torch.no_grad():
                buf_sums = [
                    f"{name}={float(buf.double().sum().item()):.9e}"
                    for name, buf in model.named_buffers()
                    if buf.numel()
                ]
                param_sums = []
                for name, p in model.named_parameters():
                    if any(k in name for k in ("embed_tokens.weight", "layers.0.", "layers.31.", "lm_head")):
                        param_sums.append(f"{name}={float(p.double().sum().item()):.9e}")
                    if len(param_sums) >= 40:
                        break
            logger.info(f"[state-checksum] {tag_str} n_buffers={len(buf_sums)}")
            for i in range(0, len(buf_sums), 8):
                logger.info(f"[state-checksum] buf: {' | '.join(buf_sums[i:i+8])}")
            for i in range(0, len(param_sums), 4):
                logger.info(f"[state-checksum] par: {' | '.join(param_sums[i:i+4])}")
        except Exception as exc:
            logger.warning(f"state-checksum failed (non-fatal): {exc}")

    def check_weights(self: Scheduler, recv_req: CheckWeightsReqInput):
        try:
            payload = self.tp_worker.model_runner.check_weights(action=recv_req.action)
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self: Scheduler, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self: Scheduler, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
