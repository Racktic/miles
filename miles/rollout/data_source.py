import abc
import copy
import logging
import os
import re
from pathlib import Path

import torch

from miles.utils.data import Dataset
from miles.utils.misc import load_function
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(
                args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
            )
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def get_samples(self, num_samples):
        # TODO further improve code
        if self.dataset is not None:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                num_samples -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:num_samples]
                self.sample_offset = num_samples
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            # Raise only when this checkpoint demonstrably tracks dataset state and the one file
            # we need is the one missing — that is the case where falling through would silently
            # replay data the run already consumed. When no state file exists at all (weights-only
            # checkpoint, or a run that never enabled rollout_global_dataset), offset 0 is the only
            # thing we can do, so keep the original info log and let the resume proceed.
            #
            # This keys off the state directory rather than start_rollout_id: by the time we get
            # here we cannot tell whether the caller passed --start-rollout-id or Megatron derived
            # it from the checkpoint (arguments.py overwrites args.start_rollout_id in several
            # places, and actor.py derives loaded_rollout_id + 1), and both are non-negative, so
            # the id alone cannot gate this.
            state_dir = os.path.join(self.args.load, "rollout")
            existing = sorted(
                int(m.group(1))
                for f in (os.listdir(state_dir) if os.path.isdir(state_dir) else [])
                if (m := re.fullmatch(r"global_dataset_state_dict_(\d+)\.pt", f))
            )
            if rollout_id is not None and rollout_id >= 0 and existing:
                # train.py saves actor weights before dataset state (and with --async-save the
                # weight save is not even synchronous), so a job killed between the two leaves
                # iter_N on disk with no state_N — a normal crash, not misuse. Refusing outright
                # would make such a checkpoint unresumable without editing miles, so provide an
                # explicit, loud opt-out rather than a dead end.
                if os.environ.get("MILES_ALLOW_MISSING_DATASET_STATE", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                ):
                    logger.warning(
                        f"Dataset state for rollout {rollout_id} is missing ({path}); "
                        "MILES_ALLOW_MISSING_DATASET_STATE is set, so data consumption RESTARTS "
                        f"FROM OFFSET 0. This run will replay data it already trained on. "
                        f"Newest state file available: rollout {existing[-1]}."
                    )
                    return
                raise RuntimeError(
                    f"Dataset state for rollout {rollout_id} is missing ({path}), but this "
                    f"checkpoint does track dataset state: {len(existing)} file(s) present, "
                    f"ids {existing[0]}..{existing[-1]}. Continuing would restart data consumption "
                    "from offset 0 and silently replay data this run already trained on. This "
                    "usually means the job died between saving weights and saving dataset state. "
                    # load() is called with start_rollout_id - 1, and state id N means "after
                    # rollout N", so the id to pass is the newest state id plus one.
                    f"Either pass --start-rollout-id {existing[-1] + 1} (state {existing[-1]} "
                    "exists), or set MILES_ALLOW_MISSING_DATASET_STATE=1 to accept restarting "
                    "data consumption from the beginning."
                )
            # No state file at all: nothing to resume from, so offset 0 is the only option — but
            # say so at warning level, since it silently changes what data the run consumes.
            logger.warning(
                f"No dataset state found at {path} and none present under {state_dir}; "
                "data consumption starts from offset 0."
            )
            return

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
