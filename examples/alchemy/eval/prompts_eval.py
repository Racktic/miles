"""TMLR-aligned prompts/rendering for the zero-shot Alchemy eval (no-summary condition).

Format mirrors Sawyer et al. (2025) Table 7-8: a `GAME STATE UPDATE` (New episode / New trial
markers, the previous action echoed as `What stone…/What potion…/Outcome stone:`, stones as
`<colour> <size> <shape> with reward <±N>`, **all 12 potion slots** with used ones shown as `None`,
and `Current trial score: N`), plus a three-part model response
(`OBSERVATION: / REASONING: / ACTION: Place stone <I> in potion <J>`).
The base system prompt (game rules) is ours — the paper only published Feature-World system prompts,
not Alchemy's; obs/action/feedback format is copied from Table 7-8.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/alchemy
from env_alchemy import _potion_desc, _stone_desc, decode_obs, parse_action  # noqa: E402

# ---------------------------------------------------------------------------------------------------
# System prompt, split to mirror Sawyer et al. (2025) Figure 9 EXACTLY so we can run the paper's two
# conditions cleanly:
#   - "no prior information"  = MAIN_PROMPT only (Figure 5a)         -> build_eval_system(prior_info=False)
#   - "prior information"     = MAIN_PROMPT + the 3 prior blocks (5b) -> build_eval_system(prior_info=True)
# MAIN_PROMPT and PRIOR_INFO_* are VERBATIM from Figure 9. FORMAT_BLOCK is ours (operational: response
# shape + action vocabulary, needed for parsing) and is identical in both conditions, so it does not
# contaminate the prior-information manipulation. The paper does NOT publish a standalone Alchemy
# response-format prompt; the OBSERVATION/REASONING/ACTION shape follows Tables 7-8.
# ---------------------------------------------------------------------------------------------------

# Figure 9 — main prompt (the "no prior information" condition), verbatim.
MAIN_PROMPT = (
    "You are playing a text-based game called Alchemy where you place stones of different shapes "
    "(round or pointy), sizes (small or large), and colors (blue or purple) into potions that change "
    "the stones' properties. The goal is to maximize the reward value of the stones, and then place "
    "them into the cauldron to increase the total score. Placing a stone in the cauldron adds the "
    "current reward value of that stone to the total score. However, there might be more rules to the "
    "game than mentioned here."
)

# Figure 9 — the three prior-information components (reward / potion pairing / causal), verbatim.
PRIOR_INFO_REWARD = "The maximum stone reward value is 15 and the minimum is -3."
PRIOR_INFO_PAIRING = (
    "There are six different potion colors, which come in three pairs: yellow/orange, red/green, and "
    "pink/turquoise. Potions in a pair have the opposite effect from each other on a given property."
)
PRIOR_INFO_CAUSAL = (
    "A stone's reward value is determined by its properties. The game is deterministic: a potion of a "
    "given color always has the same effect on a stone with given properties, and a stone with given "
    "properties always has the same reward value."
)

# Ours — operational response format (identical across both conditions).
FORMAT_BLOCK = (
    "The game is played over several trials; the GAME STATE UPDATE marks \"New episode\" and "
    "\"New trial\" boundaries, and at the start of each trial the stones and potions are refreshed.\n\n"
    "Each turn you are given a GAME STATE UPDATE and must reply in EXACTLY this format:\n"
    "OBSERVATION: <what the previous action did, if any>\n"
    "REASONING: <brief reasoning about what to do next>\n"
    "ACTION: <exactly one of>\n"
    "    Place stone <I> in potion <J>\n"
    "    Place stone <I> in the cauldron\n"
    "    End the trial"
)

# Training-target exploration hint for Qwen3-4B-Instruct-2507. This is intentionally NOT part of the
# TMLR baseline prompt; offline A/B in BASELINE_RESULTS.md showed it wakes up potion use (and therefore
# makes the WRITE reward trainable) without materially changing task score by itself.
TRAINING_EXPLORATION_HINT = (
    "Note on strategy: each potion's effect is hidden and can either INCREASE or DECREASE a stone's "
    "reward — the only way to learn it is to try. Use the first few trials to experiment: place stones "
    "into different potions and observe the Outcome stone, then exploit what you learned in later "
    "trials. Transforming a low-reward stone into a high-reward one with the right potion scores far "
    "more than just submitting the stones you start with."
)


# --- cross-trial summarization (TMLR A.4.2) ---------------------------------------------------------
# At the end of each trial the model writes a summary, conditioned on the trial's events + the previous
# summary; future trials act on that summary. The paper doesn't publish the exact summary prompt; this
# follows its description. Used in --summary mode (replace: summary REPLACES history; augment: summary
# is folded in on top of the full history).
SUMMARY_SYSTEM = (
    "You are playing Alchemy. You keep a running, structured SUMMARY of this episode's hidden chemistry "
    "across its trials. At the end of each trial you are given your PREVIOUS summary plus the trial you "
    "just played, and you output the UPDATED summary — carrying forward what you already confirmed and "
    "folding in the new trial's evidence — for use in the remaining trials."
)
# Structured two-section format, following the TMLR summary example (Potion Effects + Highest Reward
# Combination). The per-potion bullet structure resists collapsing into a vague pessimistic blob, and the
# "carry forward confirmed effects" clause counters the REWRITE drift we observed (QWEN_SUMMARY_DRIFT.md).
SUMMARY_INSTRUCTION = (
    "The trial above is over. Update your running notes on THIS episode's hidden chemistry in EXACTLY "
    "these two sections:\n\n"
    "### Potion Effects\n"
    "One bullet per potion colour you have evidence about: which stone feature it changes and in which "
    "direction, when it acts vs. has no effect, and how it changes the reward.\n\n"
    "### Highest Reward Combination\n"
    "The highest stone reward seen so far, and which stone feature combination(s) achieve it.\n\n"
    "Carry forward everything you already confirmed in your previous summary; ADD this trial's new "
    "findings and only revise an entry when THIS trial gives direct evidence against it — do NOT delete a "
    "confirmed potion effect just because a trial went badly. Output ONLY the two sections."
)

# Free-form variant: same task + same carry-forward discipline as SUMMARY_INSTRUCTION, but WITHOUT the
# fixed two-section structure / domain priors (no ### Potion Effects / ### Highest Reward scaffold) — the
# model organizes its own memory. Selected at rollout time via ALCHEMY_FREEFORM=1 (see alchemy_rollout.py).
SUMMARY_INSTRUCTION_FREEFORM = (
    "The trial above is over. Update your running notes on THIS episode's hidden chemistry. "
    "Carry forward everything you already confirmed in your previous summary; ADD this trial's new "
    "findings and only revise an entry when THIS trial gives direct evidence against it — do NOT delete a "
    "confirmed effect just because a trial went badly. "
    "Organize the notes in whatever structure you find most useful."
)


def build_eval_system(prior_info: bool = False) -> str:
    """Assemble the system prompt. prior_info=False -> Figure 5a condition; True -> Figure 5b."""
    parts = [MAIN_PROMPT]
    if prior_info:
        parts += [PRIOR_INFO_REWARD, PRIOR_INFO_PAIRING, PRIOR_INFO_CAUSAL]
    parts.append(FORMAT_BLOCK)
    return "\n\n".join(parts)


def build_training_system(prior_info: bool = False) -> str:
    """Eval-aligned system prompt plus the Qwen3-4B exploration hint used for RL rollouts."""
    return build_eval_system(prior_info=prior_info) + "\n\n" + TRAINING_EXPLORATION_HINT


EVAL_SYSTEM = build_eval_system(prior_info=False)  # default = no prior information


def render_game_state(obs: dict, last: dict | None = None, first_turn: bool = False) -> str:
    """Render the TMLR-style `GAME STATE UPDATE`.

    ``last`` (the previous action, for the echo) = ``{kind, stone, potion, outcome_desc, note}`` or
    None. ``first_turn`` adds the `New episode` marker. `New trial` comes from ``obs['new_trial']``.
    """
    d = decode_obs(obs["vec"])
    lines: list[str] = []
    if first_turn:
        lines.append("New episode")
    if obs.get("new_trial"):
        lines.append("New trial")
    if last is not None:
        if last.get("kind") in ("potion", "cauldron"):
            lines.append(f"What stone do you use? {last['stone']}")
            if last["kind"] == "potion":
                lines.append(f"What potion do you use? {last['potion']}")
            if last.get("outcome_desc"):
                lines.append(f"Outcome stone: {last['outcome_desc']}")
        if last.get("note"):
            lines.append(last["note"])
    lines.append("New observation")
    lines.append("You see the following stones:")
    any_stone = False
    for s in d["stones"]:
        if s["exists"]:
            any_stone = True
            desc, rwd = _stone_desc(s["axes"], s["value"])
            lines.append(f"{s['idx']}: {desc} with reward {rwd:+d}")
    if not any_stone:
        lines.append("(no stones remaining this trial)")
    lines.append("You see the following potions:")
    for p in d["potions"]:                       # all 12 slots; used ones -> None (keeps indices)
        lines.append(f"{p['idx']}: {_potion_desc(p['type']) if p['exists'] else 'None'}")
    lines.append(f"Current trial score: {int(round(obs.get('trial_score', 0.0)))}")
    return "\n".join(lines)


# Anchor to a line that STARTS with `ACTION:` (re.MULTILINE) and take the LAST such line — the model
# often says "action" inside its REASONING, so a plain first-match search grabs the wrong text.
_ACTION_LINE_RE = re.compile(r"^\s*ACTION\b\s*:?\s*(.+)", re.IGNORECASE | re.MULTILINE)


def parse_eval_action(text: str, obs: dict | None = None) -> dict:
    """Pull the `ACTION:` line and parse it with end-trial enabled. Falls back to the whole text if
    no `ACTION:` label (some models drop it)."""
    matches = _ACTION_LINE_RE.findall(text or "")
    region = matches[-1] if matches else (text or "")
    return parse_action(region, obs=obs, end_trial=True)
