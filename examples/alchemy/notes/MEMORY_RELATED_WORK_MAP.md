# Memory Related Work Map for Group Meeting

This is not meant to be a formal related-work section. The goal is to help frame what is close to our project and what is only a useful contrast.

## Our Framing

Our project asks:

> How can an agent turn experience into memory-guided behavior?

More specifically:

```text
past interaction experience
    -> written memory / knowledge
    -> future memory-conditioned action
    -> downstream reward
```

The key distinction is that memory is not just a context buffer. We care about whether memory changes future behavior and improves reward.

## Closest Cluster: RL-Trained Memory Operations

These are the most important works to check carefully if we want to claim novelty.

| Work | Link | Why it is close | Difference from our Alchemy setting |
|---|---|---|---|
| Memory-R2 | https://arxiv.org/abs/2605.21768 | RL framework for long-horizon memory-augmented agents; directly studies credit assignment for memory writes/updates/deletes. | More general long-horizon memory operations; our setting is a controlled symbolic environment with explicit ACT/WRITE streams and oracle-normalized trial reward. |
| Agentic Memory / AgeMem | https://arxiv.org/abs/2601.01885 | Treats memory operations such as store/retrieve/update/summarize/discard as agent actions trained with RL. | General LLM agent benchmarks rather than a controlled hidden-rule environment like Alchemy. |
| Memory-R1 | https://arxiv.org/abs/2508.19828 | Trains memory manager operations such as ADD/UPDATE/DELETE/NOOP with PPO/GRPO-style RL and downstream outcome rewards. | More retrieval/answering oriented; less focused on simultaneous environment acting and per-episode hidden dynamics. |
| Retroformer | https://arxiv.org/abs/2308.02151 | Learns a retrospective model from failed attempts and task reward; uses policy-gradient-style training. | Learns reflection/prompt improvement, not exactly a compact written world-model memory used across trials. |

Takeaway:

- This cluster is closest because memory writing/updating is trained, not just prompted.
- Our strongest distinction is the clean symbolic testbed and explicit study of WRITE reward design under measurable oracle-normalized future performance.

## Prompted / Inference-Time Experience-to-Knowledge Agents

These works are conceptually close to our high-level motivation, but usually do not train the memory writer with RL.

| Work | Link | Core idea | Comparison to us |
|---|---|---|---|
| Reflexion | https://arxiv.org/abs/2303.11366 | Agents write verbal reflections from feedback and condition future attempts on them. | Very close loop: experience -> written lesson -> future action. But memory writing is prompted/heuristic, not learned with RL. |
| ExpeL | https://arxiv.org/abs/2308.10144 | Extracts natural-language insights from successful/failed experiences and reuses them later. | Strong conceptual baseline for experience-to-rule learning. Not per-episode hidden chemistry and not RL-trained memory writing. |
| CLIN | https://arxiv.org/abs/2310.10134 | Continual language agent updates textual causal abstractions after trials, e.g. in ScienceWorld. | Good comparison for symbolic/science environments; memory updates are generated, not optimized through our ACT/WRITE RL setup. |
| Voyager | https://arxiv.org/abs/2305.16291 | Minecraft agent builds a reusable executable skill library from interaction feedback. | Similar knowledge accumulation, but the memory artifact is code/skills rather than declarative episode-specific world knowledge. |
| Generative Agents | https://arxiv.org/abs/2304.03442 | Stores observations, retrieves memories, and synthesizes reflections for planning/social behavior. | Important architecture for episodic plus reflective memory, but not reward-optimized control memory. |
| HELPER | https://arxiv.org/abs/2310.15127 | Expands an external memory of language-program examples from dialogue/corrections. | Similar deployment-time memory expansion, but aimed at instruction parsing/personalization rather than self-discovered causal knowledge. |
| Agent-Pro | https://arxiv.org/abs/2402.17574 | Reflects over trajectories and beliefs to improve policy in games. | Treats experience as policy knowledge, but not a learned WRITE stream in a controlled symbolic environment. |
| ERL: Experiential Reflective Learning | https://arxiv.org/abs/2603.24639 | Reflects on trajectories/outcomes to generate transferable heuristics. | Close to experience -> actionable heuristic; likely more cross-task and retrieval-oriented than our per-episode memory-RL setup. |

Takeaway:

- These works motivate the idea that agents should learn from experience.
- Our project asks how to train the memory writing and memory-conditioned action loop, instead of relying only on prompted reflection.

## Diagnostic / Memory Management Studies

| Work | Link | Why useful |
|---|---|---|
| How Memory Management Impacts LLM Agents | https://arxiv.org/abs/2505.16067 | Useful for framing failure modes: bad memories can cause error propagation, experience-following, or overcommitment to stale information. |
| When Continual Learning Moves to Memory | https://arxiv.org/abs/2604.27003 | Supports the idea that abstract/procedural memories can be more useful than raw trajectory storage under limited context. |

Takeaway:

- These are useful for explaining why memory quality matters, not just memory capacity.
- They support our distinction between raw history and useful knowledge.

## Classic RL / Neural Memory Background

This cluster is less LLM-specific but highly relevant to the ACT/WRITE mechanics.

| Work | Link | Memory role | Why relevant |
|---|---|---|---|
| Reinforcement Learning Neural Turing Machines | https://arxiv.org/abs/1505.00521 | Learns to interact with discrete external memory using RL. | Older analogue of learning memory operations, though not language memory. |
| Differentiable Neural Computer | https://www.nature.com/articles/nature20101 | Learns dynamic external read/write memory. | Shows memory read/write as trainable computation; mostly supervised, with some RL-style tasks. |
| Memory Q-Networks / Control of Memory, Active Perception, and Action in Minecraft | https://arxiv.org/abs/1605.09128 | Stores observations and attends over them for Q-value/action selection. | Strong precedent for memory-conditioned acting in POMDPs; writing is mostly buffer-like. |
| Neural Map | https://arxiv.org/abs/1702.08360 | Learns a structured spatial memory map for navigation. | Good example of learned memory for future control. |
| Neural Episodic Control | https://arxiv.org/abs/1703.01988 | Stores state embeddings/value estimates and queries similar memories. | Memory helps action selection, but update rule is mostly hand-designed. |
| Memory Augmented Control Networks | https://arxiv.org/abs/1709.05706 | Learns what past information to store for partially observable planning. | Mechanically close to memory for control, but not textual memory. |
| MERLIN | https://arxiv.org/abs/1803.10760 | Predictive memory supports goal-directed RL under partial observability. | Strong precedent for forming useful internal memory from experience. |
| World Models | https://arxiv.org/abs/1809.01999 | Learns latent recurrent model from experience for future control. | Compresses experience into a learned state/model, not written text. |
| RL^2 | https://arxiv.org/abs/1611.02779 | Recurrent hidden state stores fast learner state across episodes. | Relevant as implicit learned memory, but not explicit writable memory. |
| Dreamer | https://arxiv.org/abs/1912.01603 | Learns recurrent latent world model and trains policy through imagined trajectories. | Strong example of experience -> compact state -> behavior, but not textual memory writing. |

Takeaway:

- RL has a long history of learned memory for partial observability and control.
- Our project brings this question into textual LLM memory: what should the model write, and how should that text guide future decisions?

## Contrast Cluster: Memory as Context Buffer / Retrieval

These are useful for explaining what we are not primarily doing.

| Work | Link | What it optimizes | Contrast |
|---|---|---|---|
| Transformer-XL | https://arxiv.org/abs/1901.02860 | Long-range language modeling via recurrent hidden states. | Memory is a hidden-state cache, not a reward-optimized knowledge object. |
| Longformer | https://arxiv.org/abs/2004.05150 | Efficient attention over long documents. | Expands context capacity; does not learn memory writing for decision reward. |
| Compressive Transformer | https://arxiv.org/abs/1911.05507 | Compressed activation memory for long-range sequence modeling. | Memory is architectural compression, not agent-written knowledge. |
| RAG | https://arxiv.org/abs/2005.11401 | Retrieval for knowledge-intensive generation. | Retrieves facts for generation, not self-written experience knowledge for future control. |
| REALM | https://arxiv.org/abs/2002.08909 | Differentiable retrieval for masked LM / QA. | Optimizes evidence retrieval for language objectives, not future reward. |
| RETRO | https://arxiv.org/abs/2112.04426 | Retrieval-augmented next-token prediction. | Memory improves LM likelihood, not decision quality. |
| Memorizing Transformers | https://arxiv.org/abs/2203.08913 | kNN activation memory for LM performance. | Similarity-based memory, not learned memory operations. |
| MemGPT | https://arxiv.org/abs/2310.08560 | Virtual context management over memory tiers. | Clearest context-buffer foil: manages what fits in prompt, whereas we optimize behavioral utility. |

Takeaway:

- These works optimize longer context, factual grounding, or language-modeling loss.
- Our project optimizes the control value of memory: whether it improves future actions and reward.

## Suggested PPT Use

For an informal group meeting, do not show all these works.

Possible one-slide version:

**Memory as buffer**

- Transformer-XL, Compressive Transformer, RAG, MemGPT
- Goal: keep/retrieve information under context limits

**Memory as prompted reflection**

- Reflexion, ExpeL, CLIN, Voyager
- Goal: use experience-derived text or skills at inference time

**Memory as learned control object**

- RL-NTM, DNC, MERLIN, Memory-R1/R2, AgeMem
- Goal: train memory operations because they affect future behavior

**Our angle**

- Controlled symbolic environment
- Explicit ACT and WRITE streams
- Memory evaluated by future oracle-normalized reward
- Focus on WRITE reward design and exploration bottlenecks

## Short Positioning Sentence

> Prior LLM memory work often treats memory as a context-management or prompted-reflection mechanism. We study memory as a trainable control object: the model must learn what to write from experience and how to use it for future reward-seeking behavior.
