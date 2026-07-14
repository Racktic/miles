# Related Work & Competitor Survey — RL-trained memory for agents

> Purpose: position our method (RL-trained NL memory for **action** in Symbolic Alchemy, where each episode hides a latent causal rule the agent must discover by exploration and reuse across trials) against existing RL-memory work.
>
> ⚠️ **Verification status.** Only the three anchors (MEM1, MemPO, MemAgent) were read in full text and verified. Everything in the survey below was surfaced by web search and is **NOT yet verified** — several names/IDs look suspiciously on-the-nose (e.g. HiMPO, Mem-T) and some IDs are near-future (2603/2606/2512). **Verify each paper exists and says what is claimed before citing.** The true-competitor trio (MemoPilot / MAGE / LaMer) should be read in full before final positioning.

---

## 1. Related Work (draft prose)

**RL-trained natural-language memory for long-horizon and long-context agents.**
A growing line trains language models, with reinforcement learning, to write a compact natural-language memory that replaces the full interaction history. MemAgent [2507.02259] reads a long document in fixed chunk order and overwrites a fixed-size summary at each step under a final-answer reward; because the reading order is fixed, the memory is a feed-forward compressor that conditions only the terminal readout. MEM1 [2506.15841] and MemPO [2603.00680] extend this to interactive multi-turn agents: MEM1 consolidates an "internal state" and prunes old turns under a PPO outcome reward, while MemPO trains a rolling `<mem>` summary with GRPO plus an auxiliary memory-level reward measuring whether the memory suffices to answer. A cluster of recent methods refine the same recipe — operation-level memory editing (Memory-R1 [2508.19828], Memory-as-Action [2510.12635]), learned memory construction (Mem-α [2509.25911]), write-and-read external indices (Memex [2603.04257]), and denser or less-entangled memory rewards (Mem-T [2601.23014], HiMPO [2606.16285]). Across this line, the memory operates in a **retention** regime: the task-relevant information already exists in the environment and is directly observable on query, so the learned memory is a working-state ledger that compresses given facts for efficiency and length-robustness. Our setting differs at the level of the problem: the information our memory must hold does not exist to be retrieved — it is a hidden latent rule recoverable only by **inducing** it from action–outcome experiments — so our memory must abstract a generalizable strategy rather than retain observed facts.

**RL-optimized memory for action under unknown dynamics.**
Closest to our goal are methods that optimize a natural-language memory **for acting** in tasks whose dynamics must be discovered. MemoPilot [2606.08656] trains, with multi-turn GRPO, a memory generator that maintains diagnoses, confidence-weighted beliefs, and next-game guidance, credited by subsequent outcomes, to exploit opponent patterns at test time; MAGE [2603.03680] and LaMer [2512.16848] use meta-RL to optimize cross-episode natural-language reflections for strategic exploration and rapid in-context adaptation in environments such as ALFWorld, WebShop, Sokoban, and card games. We share with this line the use of RL-optimized, inspectable NL memory to improve future **actions** rather than answers. We differ in the structure of what must be learned: Symbolic Alchemy resamples a **latent causal rule** each episode, making the core problem system identification of a compositional transition function; our memory must encode a rule that predicts the outcome of **untried** actions and is **shared across trials within an episode**, which we evaluate through an explicit within-episode improvement curve. We further reward the acting policy for **exploration that produces informative experience**, and co-train acting and writing rather than optimizing writing alone.

**Prompted experience distillation and skill libraries.**
Reflexion [2303.11366], ExpeL [2308.10144], Voyager [2305.16291], AutoManual [2405.16247], and AutoGuide turn experience into reusable verbal insights, rules, or executable skills, and thus also target an **abstraction** memory. Unlike these, we do not rely on a fixed prompting pipeline: we optimize the memory-writing policy with reinforcement learning against downstream reward.

**In-context and meta-reinforcement learning.**
Our framing connects to in-context RL, where adaptation to an unknown task emerges within the context window — e.g., Algorithm Distillation [2210.14215] and the human-timescale adaptive agent AdA [2301.07608] — and to the Symbolic Alchemy benchmark [2102.02926, 2112.08360], whose per-episode latent causal structure we adopt. These works realize in-context adaptation through implicit (latent or context-window) memory; we instead make the learned knowledge an explicit, inspectable natural-language rule that is itself optimized to drive action.

---

## 2. The three anchors (verified by full read)

| Paper | id | Task | Memory | RL | Reward | Regime |
|---|---|---|---|---|---|---|
| **MEM1** | 2506.15841 | multi-turn agent QA (2–16 indep. objectives), WebShop | `<IS>` internal state, consolidate+prune, constant-size | PPO, joint policy | final outcome only | retention (facts + per-Q progress + reasoning); Fig 5(a) keeps "Memory for Q1/Q2" separately |
| **MemPO** | 2603.00680 | retrieval QA / deep research | rolling `<mem>` summary; next step sees only memory | GRPO | trajectory + **memory-level P[ans\|mem]** | retention; prompt **forbids** inferred knowledge in memory (fact-ledger by design) |
| **MemAgent** | 2507.02259 | long-**document** QA | fixed-size summary, overwrite per chunk | Multi-Conv DAPO | final answer | **no decision loop** (fixed read order) → pure compression |

Concrete memory examples (verbatim):
- MemPO `<mem>`: *"DJ Khaled graduated from Dr. Phillips High School. Need to find the Olympic diver from this school."* → retrieved-facts + pending sub-question.
- MEM1 `<IS>`: *"...yoga originated in India. Now, we need to find the capital of India..."* → found fact + pending sub-goal + reasoning.
- **Ours (contrast):** *"Yellow can increase value for blue round stones; try it before cashing out."* → generalizable rule that predicts untried combos and changes the action.

---

## 3. Survey of related works (search-surfaced — VERIFY before citing)

### Group A — retention neighbors (RL + compress given history into NL memory) — same regime as anchors
| Method | id | RL writes memory? | Memory type | Closeness to anchors |
|---|---|---|---|---|
| Memory-as-Action / MemAct | 2510.12635 | yes | retention (edit-actions) | HIGH |
| Mem-α | 2509.25911 | yes | retention (build memory) | HIGH |
| Memex(RL) | 2603.04257 | yes | retention (write+read index) | HIGH (positions vs summary-only) |
| Memory-R1 | 2508.19828 | yes | retention (ADD/UPDATE/DELETE bank) | HIGH |
| Mem-T (densify reward) | 2601.23014 | yes | retention | HIGH (MemPO follow-up) |
| HiMPO (hindsight memory PO) | 2606.16285 | yes | retention | HIGH (near-clone of MemPO) |
| Agentic Memory / AgeMem | 2601.01885 | yes | retention | HIGH |
| Look Back to Reason Forward | 2509.23040 | partial | retention (revisitable) | MED-HIGH (MemAgent contrast) |
| Learn to Memorize | 2508.16629 | partial | retention | MED |

→ Our retention-vs-abstraction distinction covers this whole group; not a novelty threat.

### Group B — TRUE COMPETITORS (RL-optimized NL memory for ACTION under unknown dynamics) ★
| Method | id | What | Why it's the real competitor |
|---|---|---|---|
| **MemoPilot** (From Player to Master) | 2606.08656 | multi-turn GRPO trains NL belief/strategy memory; credited by next-game outcome; exploits opponent at test time | **Closest**: RL-optimized NL memory for action under unknown dynamics |
| **MAGE** | 2603.03680 | meta-RL, cross-episode NL reflection, learning-progress reward; exploration + opponent adaptation | very high |
| **LaMer** (Meta-RL Induces Exploration) | 2512.16848 | meta-RL trains post-episode NL reflection for in-context test-time adaptation | very high |
| Agent-Pro | 2402.17574 | NL policy/belief memory for action | high on idea, but **prompted, not RL** (beatable baseline) |

→ **None appear to use Symbolic Alchemy / latent-causal-structure.** Our differentiation must be the **regime specifics** (latent causal rule discovery / system identification / structured world-model memory / across-trial shared rule + learning curve), NOT the generic claim "RL NL memory for action," which this trio already occupies.

### Group C — in-context / meta-RL foundations (no LLM-written NL memory)
Algorithm Distillation [2210.14215]; AdA / XLand [2301.07608]; Symbolic Alchemy benchmark [2102.02926, 2112.08360]. Conceptual ancestors of the exploration/unknown-dynamics framing; Alchemy is our evaluation substrate.

### Group D — prompted experience-distillation / skill libraries (abstraction, but NOT RL)
Reflexion [2303.11366], ExpeL [2308.10144], Voyager [2305.16291], AutoManual [2405.16247], AutoGuide. Prompted baselines an RL-memory paper would beat.

---

## 4. Take-away for positioning
1. **vs the three anchors + Group A:** clean — they are *retention* over given info; we are *abstraction* of discovered latent dynamics. (Two-tier argument: behavioral loop separates us from MemAgent/RAG; exploration-of-unknown-dynamics separates us from MEM1/MemPO.)
2. **vs Group B (the ones that matter):** we CANNOT claim "first RL-trained NL memory for action / in-context meta-learning" — MemoPilot/MAGE/LaMer already do it. Reposition on: **latent compositional causal rule (system identification), structured world-model memory, across-trial shared structure with an explicit within-episode improvement curve, and an exploration reward for ACT.** Must cite & explicitly distinguish this trio.
3. **vs Group D:** we train the writing with RL; they prompt.

**Next action:** full-read-verify MemoPilot (2606.08656), MAGE (2603.03680), LaMer (2512.16848) before locking the related-work, since they are the only works that could undercut the core novelty.


Sharing two meta-RL-for-LLM-agent papers. MAGE is a direct follow-up to LaMer and shares most of its formulation — so it's cleanest to lay out the common setup first, then the few places they actually diverge.

## Shared formulation

Both treat a *trial* as a sequence of $N$ episodes $\mathcal{T} = (\tau^{(0)}, \dots, \tau^{(N-1)})$ against a fixed task/opponent. After each episode the agent generates a natural-language **reflection**, and the policy adapts **in-context** (no gradient update) by conditioning the next episode on accumulated history + reflections. Both use a per-action **step-wise return** that mixes a within-episode term with a $\gamma_{traj}$-discounted cross-episode term:

$$G_t^{(n)} = \underbrace{\sum_{l=t}^{T-1} \gamma_{step}^{l-t}\, r_l^{(n)}}_{\text{within-episode}} + \underbrace{\sum_{m=n+1}^{N-1} \gamma_{traj}^{m-n}\, G_0^{(m)}}_{\text{cross-episode}}$$

Both optimize with GiGPO on Qwen3-4B, $N=3$, $\gamma_{traj}=0.6$, meta group size 8 vs. RL group size 24 (matched trajectory count). MAGE's step-wise return is explicitly "Inspired by LaMer."

## Where they diverge

**1. What the reward feeding the return is.** This is the core difference.
- LaMer: each episode contributes its **absolute** return. Objective: $J(\theta) = \mathbb{E}\big[\sum_n \gamma_{traj}^n \sum_t \gamma_{step}^t r_t^{(n)}\big] = \mathbb{E}\big[G_0^{(0)}\big]$ — scores the **sum over all $N$ attempts**.
- MAGE: each episode contributes a **differential** meta-reward $R_n = R(\tau_n) - R(\tau_{n-1})$, $R(\tau_0)\equiv 0$. Objective $\max_\theta \mathbb{E}\big[\sum_{n=1}^N R_n\big]$ telescopes to $R(\tau_N)$ — scores **only the final attempt**, treating earlier episodes as pure exploration ("slow-start, high-finish").
- Note: the objective telescopes to $R(\tau_N)$, but the differential signal is still injected per-step via the shared $\gamma_{traj}$ return above — so *equivalent objective ≠ equivalent gradient*. The real empirical difference rests on MAGE's reward-design ablation.

**2. Whether reflection is trained.**
- LaMer explicitly trains the reflection step (reflection tokens enter the gradient, supervised by the next episode's reward).
- MAGE's main-text loss (Eq. 9) sums over **action tokens only**; reflection appears solely as context input, not in the loss. (Per the main-text formulation; token-mask details would need the code.)

**3. Single- vs multi-agent, and MAGE's extra machinery.**
- LaMer: single-agent only — Sokoban / MineSweeper / Webshop / ALFWorld.
- MAGE: keeps Sokoban / Webshop / ALFWorld, drops MineSweeper, adds two games (Tic-Tac-Toe / Kuhn Poker) for *strategic exploitation* of opponents. To support this it adds **population-based training** (an opponent pool) + **agent-specific advantage normalization** — these are its substantive contributions beyond the reward change.

**Caveat:** MAGE's reported LaMer numbers are its own re-runs under a final-episode protocol, not directly comparable to LaMer's original pass@3.