"""Human-authored prompts for the ARC-AGI-3 memory agent.

This file isolates all hand-written prompt text so it can be iterated on without touching the
rollout logic. ``arc_rollout.py`` imports ``DEFAULT_SYSTEM`` and ``render_user_text`` from here.

- ``DEFAULT_SYSTEM``: the fixed system prompt (game rules, action space, the memory-only-context
  contract, and the required ``<reasoning>…</reasoning><memory>…</memory><action>…</action>`` format).
  The ``<reasoning>`` block is a per-turn scratchpad that is NOT carried forward (parser drops it);
  only ``<memory>`` accumulates. This keeps memory from being polluted by the model's reasoning.
- ``render_user_text(memory, last, obs)``: the per-turn user message text (previous memory +
  last-action result/diff + current state + available actions). The current grid IMAGE is attached
  separately by the rollout, not here.
"""
from __future__ import annotations

DEFAULT_SYSTEM = """You are playing an unfamiliar turn-based grid puzzle game (ARC-AGI-3). \
The screen is a 64x64 grid of cells, each an integer colour 0-15, shown to you as an image with \
x=column (0-63) and y=row (0-63) coordinate labels.

Actions: RESET, ACTION1..ACTION7. ACTION1-5 and ACTION7 take no arguments. ACTION6 is a click and \
REQUIRES x and y in [0,63]. Each turn only the actions listed under "Available actions" are legal. \
Your goal is to increase levels_completed and reach state=WIN in as few actions as possible; avoid GAME_OVER.

You do NOT see the history of past turns. Each turn you get only: your own MEMORY, the result of \
your last action, and the current screen.

Respond every turn in this format:
<reasoning>
Think here: read the screen, digest what your last action did, analyze what you have learned, and decide your next action. This is \
a private scratchpad — it is not saved and you will not see it next turn.
</reasoning>
<memory>
Your notes, carried to your next turn — your only long-term record of what you've learned.
</memory>
<action>ACTIONk</action>

For a click use: <action>ACTION6 x=COL y=ROW</action>"""


def render_user_text(memory: str, last: dict | None, obs: dict) -> str:
    """Build the per-turn user text: prior memory + last transition + current state + available actions."""
    parts = []
    if memory:
        parts.append("# Your MEMORY (written on previous turns):\n" + memory)
    else:
        parts.append("# Your MEMORY: (empty — this is the first turn)")
    if last is not None:
        parts.append(f"# Result of your last action ({last['action']}):\n{last['diff']}")
    parts.append(f"# Current screen: state={obs['state']}, levels_completed={obs['levels']}")
    avail = obs.get("available") or []
    parts.append("# Available actions this turn: " + (", ".join(f"ACTION{a}" for a in avail) or "(none listed)"))
    parts.append("The attached image is the current 64x64 grid. Now write your <reasoning>, then your <memory>, then your <action>.")
    return "\n\n".join(parts)


# ======================================================================================
# Two-phase (ACT -> REWRITE) interface.  ⚠ PROMPT TEXT BELOW IS PLACEHOLDER — wording TBD.
# The rollout calls: ACT uses SYSTEM_ACT + render_act_text(); REWRITE uses SYSTEM_REWRITE
# + render_rewrite_text(). MEMORY_TEMPLATE is M_{-1} (the memory carried into turn 0).
# Only the *structure* (which fields go where) is fixed here; the wording is iterated later.
# ======================================================================================

# Initial structured memory carried into turn 0 (the 4 sections the REWRITE step maintains).
MEMORY_TEMPLATE = """## Observed action effects (ONE line per action; carry every line forward — only edit the action you took this turn)
- ACTION1: untested
- ACTION2: untested
- ACTION3: untested
- ACTION4: untested
- ACTION5: untested
- ACTION6: untested
- ACTION7: untested

## Hypothesis about the goal / win condition
(unknown)

## Current plan
(probe each action to learn its effect)

## Uncertain / to test
(everything)"""

# --- ACT phase: decide ONE action from memory + current screen ---  [PLACEHOLDER wording]
SYSTEM_ACT = """You are playing an unfamiliar turn-based grid puzzle (ARC-AGI-3): a 64x64 grid \
shown as an image (x=column, y=row, 0-63). Actions: RESET, ACTION1..ACTION7. ACTION1-5 and ACTION7 \
take no arguments; ACTION6 is a click and REQUIRES x,y in [0,63]. Only the listed available actions \
are legal. Goal: increase levels_completed and reach state=WIN in as few actions as possible; avoid \
GAME_OVER. You are given your MEMORY (insights you accumulated) and the current screen — decide ONE action.

Respond in this format:
<reasoning>Brief: given your memory and the screen, why this action.</reasoning>
<action>ACTIONk</action>

For a click use: <action>ACTION6 x=COL y=ROW</action>"""

# --- REWRITE phase: update memory from the just-observed transition ---  [PLACEHOLDER wording]
SYSTEM_REWRITE = """You are playing an unfamiliar turn-based grid puzzle (ARC-AGI-3). MEMORY is your \
ONLY record across turns — you never see past frames, only MEMORY. This step you UPDATE memory from \
what just happened: you are shown your current memory, the action you just took, the screen BEFORE and \
AFTER it, and the EXACT change it caused. Rewrite your memory into an updated version, keeping EXACTLY \
these sections:
## Observed action effects
## Hypothesis about the goal / win condition
## Current plan
## Uncertain / to test

Integrate what this turn revealed; REVISE or DELETE anything now wrong — do NOT merely append. Output \
ONLY the updated memory."""


def render_act_text(memory: str, obs: dict, grid_matrix: str) -> str:
    """ACT user text: memory + state + the EXACT grid matrix + available actions (image attached too)."""
    parts = [
        "# YOUR MEMORY\n" + (memory or "(empty)"),
        f"# Current screen: state={obs['state']}, levels_completed={obs['levels']}",
        "# Grid matrix (exact cell values; y=row, x=col, both 0-63):\n" + grid_matrix,
    ]
    avail = obs.get("available") or []
    parts.append("# Available actions this turn: "
                 + (", ".join(f"ACTION{a}" for a in avail) or "(none listed)"))
    parts.append("The attached image shows the same 64x64 grid as the matrix above. "
                 "Write your <reasoning>, then your <action>.")
    return "\n\n".join(parts)


def render_rewrite_text(memory: str, last: dict, before_matrix: str, after_matrix: str) -> str:
    """REWRITE user text: memory + the action + BEFORE/AFTER grid matrices (two images attached too).

    ``last`` = {action, diff, changed(bool), state, levels}. Both the EXACT matrices and the two images
    are given: matrices for precise per-cell values/coordinates, images for overall shape. The computed
    text diff is intentionally NOT shown (it was found to confuse); ``last['diff']`` is still recorded
    in the trajectory for offline review.
    """
    changed = "CHANGED" if last.get("changed") else "did NOT change (no-op)"
    parts = [
        "# CURRENT MEMORY\n" + (memory or "(empty)"),
        (f"# WHAT JUST HAPPENED\nYou took {last.get('action')}. The screen {changed}. "
         f"state={last.get('state')}, levels_completed={last.get('levels')}."),
        "# Grid matrix BEFORE the action (exact values; y=row, x=col, 0-63):\n" + before_matrix,
        "# Grid matrix AFTER the action:\n" + after_matrix,
        "Two images are also attached (BEFORE then AFTER). Use the matrices for exact per-cell "
        "values/coordinates and the images for overall shape; compare BEFORE vs AFTER to see what your "
        "action did, then rewrite your memory now.",
    ]
    return "\n\n".join(parts)


# ======================================================================================
# Object-only variants (NO image, NO matrix) — used by arc_rollout_objects.py. Pure-text
# object-centric state; these are ADDITIONS that do not touch the VLM/matrix functions above.
# ======================================================================================

SYSTEM_ACT_OBJ = """You are playing an unfamiliar turn-based grid puzzle (ARC-AGI-3): a 64x64 grid \
given to you as a LIST OF OBJECTS (each with color, size, bounding box, center, shape). \
COORDINATES: the origin (0,0) is the TOP-LEFT corner; x=column increases to the RIGHT, y=row \
increases DOWNWARD (0-63). So a smaller y is HIGHER on screen — moving UP means y DECREASES (dy<0), \
DOWN means y increases (dy>0), LEFT means dx<0, RIGHT means dx>0. \
Action names (ACTION1..ACTION7) are OPAQUE labels — they carry NO inherent direction or meaning; rely \
only on what your MEMORY says each action actually did (observed dx/dy), never on what the name sounds like. \
Actions: RESET, ACTION1..ACTION7. ACTION1-5 and ACTION7 take no arguments; ACTION6 is \
a click and REQUIRES x,y in [0,63]. Only the listed available actions are legal. Goal: increase \
levels_completed and reach state=WIN in as few actions as possible; avoid GAME_OVER. You are given \
your MEMORY (insights you accumulated) and the current objects — decide ONE action.

Respond in this format:
<reasoning>Brief: given your memory and the objects, why this action.</reasoning>
<action>ACTIONk</action>

For a click use: <action>ACTION6 x=COL y=ROW</action>"""

SYSTEM_REWRITE_OBJ = """You are playing an unfamiliar turn-based grid puzzle (ARC-AGI-3). MEMORY is \
your ONLY record across turns — you never see past frames, only MEMORY. This step you UPDATE memory \
from what just happened: you are shown your current memory, the action you just took, and the object \
lists BEFORE and AFTER it. COORDINATES: origin (0,0) is the TOP-LEFT corner; x increases RIGHT, \
y increases DOWNWARD, so UP = y DECREASES (dy<0), DOWN = y increases (dy>0), LEFT = dx<0, RIGHT = dx>0 \
— use directions consistently with this. Action names (ACTION1..ACTION7) are OPAQUE labels with NO \
inherent direction or meaning; record each action's effect ONLY by the observed dx/dy, never assume the \
name implies a direction. Object ids are STABLE across turns (an object keeps its id, \
so you can track it); an object marked [OCCLUDED] is hidden behind another object but still there \
(its position is predicted).

Respond in EXACTLY this format:
<reasoning>
Private scratchpad — DISCARDED, never saved. Do ALL your thinking here: compare the BEFORE and AFTER \
object lists, work out what the action did, decide what to record.
</reasoning>
<memory>
## Observed action effects (ONE line per action — see PER-ACTION TABLE rule)
## Hypothesis about the goal / win condition
## Current plan
## Uncertain / to test
</memory>

RULES for <memory> — this is your ONLY permanent record, treat it carefully:
- PER-ACTION TABLE: keep "## Observed action effects" as ONE line per action (ACTION1..ACTION7). You \
took exactly ONE action this turn — update ONLY that action's line with what you just observed, and copy \
EVERY other action's line VERBATIM from the current memory. Never shrink the table to only this turn's \
action; never drop a tested action's finding.
- ACCUMULATE, do NOT rewrite from scratch. KEEP every earlier finding that is still correct; only ADD \
new findings and FIX what is now proven wrong. Silently dropping a still-correct earlier finding is a \
FAILURE — your knowledge must grow across turns, not reset each turn.
- CONCISE: NO narration, NO hedging words (possibly / might / \
suggesting), NO step-by-step reasoning — all of that goes in <reasoning>, never in <memory>.
Output ONLY <reasoning>...</reasoning> then <memory>...</memory>."""


def render_act_objects_text(memory: str, obs: dict, objects_text: str) -> str:
    """ACT user text (object-only, no image): memory + state + object list + available actions."""
    parts = [
        "# YOUR MEMORY\n" + (memory or "(empty)"),
        f"# Current screen: state={obs['state']}, levels_completed={obs['levels']}",
        "# Objects on screen (x=col, y=row, 0-63):\n" + objects_text,
    ]
    avail = obs.get("available") or []
    parts.append("# Available actions this turn: "
                 + (", ".join(f"ACTION{a}" for a in avail) or "(none listed)"))
    parts.append("Write your <reasoning>, then your <action>.")
    return "\n\n".join(parts)


def render_rewrite_objects_text(memory: str, last: dict, before_objects: str, after_objects: str) -> str:
    """REWRITE user text (object-only, no image): memory + action + BEFORE/AFTER object lists."""
    changed = "CHANGED" if last.get("changed") else "did NOT change (no-op)"
    parts = [
        "# CURRENT MEMORY\n" + (memory or "(empty)"),
        (f"# WHAT JUST HAPPENED\nYou took {last.get('action')}. The screen {changed}. "
         f"state={last.get('state')}, levels_completed={last.get('levels')}."),
        "# Objects BEFORE the action (x=col, y=row, 0-63):\n" + before_objects,
        "# Objects AFTER the action:\n" + after_objects,
        "Compare the BEFORE and AFTER object lists to see what your action did, "
        "then rewrite your memory now.",
    ]
    return "\n\n".join(parts)
