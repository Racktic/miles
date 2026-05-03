"""
YAML experiment config loader for miles.

Allows specifying experiment configurations in YAML files instead of (or in
addition to) CLI arguments.  The YAML file uses nested sections for
readability, but every leaf key must match an argparse ``dest`` name exactly
(underscored, e.g. ``rollout_batch_size``).

Precedence (highest → lowest):
    CLI arguments  >  YAML config  >  argparse defaults

Usage::

    python train.py --config configs/qwen3-4b-grpo.yaml          # all from YAML
    python train.py --config configs/qwen3-4b-grpo.yaml --lr 2e-6 # override lr
"""

import copy
import logging
import os

import yaml

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_includes(config: dict, base_dir: str) -> dict:
    """Resolve ``_includes`` (list of YAML paths) by loading and merging them
    in order, then overlaying the current file's own keys on top.

    Included paths are relative to *base_dir* (the directory of the file that
    contains the ``_includes`` directive) unless they are absolute.
    """
    includes = config.pop("_includes", None)
    if not includes:
        return config

    if isinstance(includes, str):
        includes = [includes]

    merged: dict = {}
    for inc_path in includes:
        inc_path = os.path.expanduser(inc_path)
        if not os.path.isabs(inc_path):
            inc_path = os.path.join(base_dir, inc_path)
        inc_path = os.path.normpath(inc_path)
        inc_config = load_yaml(inc_path)
        merged = _deep_merge(merged, inc_config)

    # The current file's keys override anything from includes.
    merged = _deep_merge(merged, config)
    return merged


def load_yaml(path: str) -> dict:
    """Load a YAML file, resolve ``_includes``, and return the raw dict."""
    path = os.path.expanduser(path)
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    base_dir = os.path.dirname(os.path.abspath(path))
    config = _resolve_includes(config, base_dir)
    return config


def flatten_yaml(config: dict) -> dict:
    """Flatten a (possibly nested) config dict to a single-level dict.

    Section keys (the top-level grouping keys like ``cluster``, ``rollout``,
    ``algorithm``, …) are stripped.  The leaf key names are expected to match
    argparse ``dest`` names exactly.

    Only one level of nesting is unwrapped – if a leaf value is itself a dict
    (e.g. for JSON-typed args like ``train_env_vars``), it is kept as-is.
    """
    flat: dict = {}
    for key, value in config.items():
        if isinstance(value, dict):
            # Check if this looks like a "section" (all-or-mostly scalar leaves)
            # vs a single dict-valued argument (e.g. train_env_vars).
            # Heuristic: if *any* leaf is a dict we treat it as a section that
            # itself may contain dict-valued args (only 1 level deep).
            for sub_key, sub_value in value.items():
                if sub_key in flat:
                    logger.warning(
                        "Duplicate key '%s' in YAML config (appears in multiple sections). "
                        "Later value will override earlier one.",
                        sub_key,
                    )
                flat[sub_key] = sub_value
        else:
            flat[key] = value
    return flat


def apply_yaml_to_parser(parser, yaml_path: str) -> dict:
    """Load *yaml_path*, flatten it, apply values as parser defaults, and
    relax ``required`` for any arg whose value is supplied by the YAML.

    Returns the flat config dict for reference.
    """
    config = load_yaml(yaml_path)
    flat = flatten_yaml(config)

    # argparse set_defaults does NOT apply `type` functions to defaults.
    # For args with custom type parsers (e.g. moe_freq_type), we need to
    # manually apply the type function so the value is properly converted.
    dest_to_action = {a.dest: a for a in parser._actions}
    for key, value in flat.items():
        action = dest_to_action.get(key)
        if action and action.type and isinstance(value, str):
            # Strip spaces from list-pattern strings (e.g. "[1, 1, 0]" -> "[1,1,0]")
            # Megatron's _eval_pattern regex disallows spaces.
            if '[' in value:
                value = value.replace(' ', '')
            try:
                flat[key] = action.type(value)
            except (ValueError, TypeError):
                pass  # leave as-is, let argparse report the error later

    # Apply as defaults so that explicit CLI args still win.
    parser.set_defaults(**flat)

    # Relax ``required`` for args satisfied by the YAML.
    for action in parser._actions:
        if action.required and action.dest in flat:
            action.required = False

    logger.info("Loaded YAML config from %s (%d keys)", yaml_path, len(flat))
    return flat
