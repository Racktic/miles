import argparse
import os
import sys
import tempfile
from unittest.mock import patch

import pytest
import yaml

from miles.utils.yaml_config import apply_yaml_to_parser, flatten_yaml, load_yaml


class TestFlattenYaml:
    def test_flat_passthrough(self):
        cfg = {"lr": 1e-6, "batch_size": 32}
        assert flatten_yaml(cfg) == cfg

    def test_nested_sections_flattened(self):
        cfg = {
            "cluster": {"actor_num_nodes": 1, "colocate": True},
            "data": {"prompt_data": "/path/to/data.jsonl"},
        }
        assert flatten_yaml(cfg) == {
            "actor_num_nodes": 1,
            "colocate": True,
            "prompt_data": "/path/to/data.jsonl",
        }

    def test_top_level_scalars_preserved(self):
        cfg = {
            "lr": 1e-6,
            "cluster": {"actor_num_nodes": 2},
        }
        assert flatten_yaml(cfg) == {"lr": 1e-6, "actor_num_nodes": 2}

    def test_dict_values_kept_as_is(self):
        """Dict-valued leaves (e.g. train_env_vars) should not be flattened."""
        cfg = {
            "train": {
                "train_env_vars": {"CUDA_DEVICE_MAX_CONNECTIONS": "1"},
            }
        }
        flat = flatten_yaml(cfg)
        assert flat["train_env_vars"] == {"CUDA_DEVICE_MAX_CONNECTIONS": "1"}


class TestLoadYaml:
    def test_basic_load(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump({"lr": 0.001, "batch_size": 8}))
        cfg = load_yaml(str(p))
        assert cfg == {"lr": 0.001, "batch_size": 8}

    def test_includes(self, tmp_path):
        model_cfg = tmp_path / "model.yaml"
        model_cfg.write_text(yaml.dump({"model": {"hidden_size": 2560}}))

        main_cfg = tmp_path / "main.yaml"
        main_cfg.write_text(yaml.dump({
            "_includes": ["model.yaml"],
            "cluster": {"actor_num_nodes": 1},
        }))

        cfg = load_yaml(str(main_cfg))
        assert cfg["model"]["hidden_size"] == 2560
        assert cfg["cluster"]["actor_num_nodes"] == 1

    def test_includes_override(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"data": {"lr": 0.01, "batch_size": 32}}))

        main = tmp_path / "main.yaml"
        main.write_text(yaml.dump({
            "_includes": ["base.yaml"],
            "data": {"lr": 0.001},
        }))

        cfg = load_yaml(str(main))
        assert cfg["data"]["lr"] == 0.001
        assert cfg["data"]["batch_size"] == 32


class TestApplyYamlToParser:
    def test_defaults_set(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"data": {"batch_size": 64, "lr": 0.001}}))

        parser = argparse.ArgumentParser()
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=0.01)

        apply_yaml_to_parser(parser, str(p))

        with patch.object(sys, "argv", ["test"]):
            args = parser.parse_args([])
        assert args.batch_size == 64
        assert args.lr == 0.001

    def test_cli_overrides_yaml(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"data": {"batch_size": 64}}))

        parser = argparse.ArgumentParser()
        parser.add_argument("--batch-size", type=int, default=32)

        apply_yaml_to_parser(parser, str(p))
        args = parser.parse_args(["--batch-size", "128"])
        assert args.batch_size == 128

    def test_required_relaxed(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"rollout_batch_size": 64}))

        parser = argparse.ArgumentParser()
        parser.add_argument("--rollout-batch-size", type=int, required=True)

        apply_yaml_to_parser(parser, str(p))
        # Should not raise even though no CLI arg is given
        args = parser.parse_args([])
        assert args.rollout_batch_size == 64

    def test_boolean_store_true(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"colocate": True}))

        parser = argparse.ArgumentParser()
        parser.add_argument("--colocate", action="store_true", default=False)

        apply_yaml_to_parser(parser, str(p))
        args = parser.parse_args([])
        assert args.colocate is True

    def test_list_args(self, tmp_path):
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump({"eval_prompt_data": ["aime", "/path/to/data.jsonl"]}))

        parser = argparse.ArgumentParser()
        parser.add_argument("--eval-prompt-data", nargs="+", default=None)

        apply_yaml_to_parser(parser, str(p))
        args = parser.parse_args([])
        assert args.eval_prompt_data == ["aime", "/path/to/data.jsonl"]
