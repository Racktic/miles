"""Small Miles type shim for running logic tests outside the training image.

The real Miles classes are used in the Miles/Apptainer environment.  The host
checkout intentionally lacks torch, so these tests install only the interfaces
needed by the rollout unit test when torch cannot be imported.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


if importlib.util.find_spec("torch") is None:
    miles = types.ModuleType("miles")
    miles.__path__ = []
    rollout_package = types.ModuleType("miles.rollout")
    rollout_package.__path__ = []
    utils_package = types.ModuleType("miles.utils")
    utils_package.__path__ = []
    base_types = types.ModuleType("miles.rollout.base_types")
    types_module = types.ModuleType("miles.utils.types")
    http_module = types.ModuleType("miles.utils.http_utils")
    metric_module = types.ModuleType("miles.utils.metric_utils")
    tracking_module = types.ModuleType("miles.utils.tracking_utils")

    @dataclass
    class Sample:
        class Status(Enum):
            PENDING = "pending"
            COMPLETED = "completed"
            TRUNCATED = "truncated"
            ABORTED = "aborted"
            FAILED = "failed"

        group_index: int | None = None
        index: int | None = None
        prompt: Any = ""
        tokens: list[int] = field(default_factory=list)
        response: str = ""
        response_length: int = 0
        reward: float | None = None
        loss_mask: list[int] | None = None
        rollout_log_probs: list[float] | None = None
        weight_versions: list[str] = field(default_factory=list)
        status: Any = Status.PENDING
        metadata: dict[str, Any] = field(default_factory=dict)

        def validate(self):
            assert len(self.tokens) >= self.response_length
            if self.loss_mask is not None:
                assert len(self.loss_mask) == self.response_length
            if self.rollout_log_probs is not None:
                assert len(self.rollout_log_probs) == self.response_length

        def to_dict(self):
            value = asdict(self)
            value["status"] = self.status.value
            return value

        @classmethod
        def from_dict(cls, value):
            payload = dict(value)
            payload["status"] = cls.Status(payload["status"])
            return cls(**payload)

    @dataclass(frozen=True)
    class GenerateFnInput:
        state: Any
        sample: Sample
        sampling_params: dict[str, Any]
        evaluation: bool

        @property
        def args(self):
            return self.state.args

    @dataclass(frozen=True)
    class GenerateFnOutput:
        samples: Sample | list[Sample]

    async def post(*args, **kwargs):
        raise AssertionError("host-unit shim HTTP function should be monkeypatched")

    def compute_rollout_step(args, rollout_id):
        del args
        return rollout_id

    def log(*args, **kwargs):
        del args, kwargs

    base_types.GenerateFnInput = GenerateFnInput
    base_types.GenerateFnOutput = GenerateFnOutput
    types_module.Sample = Sample
    http_module.post = post
    metric_module.compute_rollout_step = compute_rollout_step
    tracking_module.log = log
    utils_package.tracking_utils = tracking_module
    sys.modules.update(
        {
            "miles": miles,
            "miles.rollout": rollout_package,
            "miles.rollout.base_types": base_types,
            "miles.utils": utils_package,
            "miles.utils.types": types_module,
            "miles.utils.http_utils": http_module,
            "miles.utils.metric_utils": metric_module,
            "miles.utils.tracking_utils": tracking_module,
        }
    )
