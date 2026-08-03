# MetaMetaController: ARC-AGI-3 Hybrid Agent Roadmap

## Goal

Build a practical ARC-AGI-3 agent by combining three complementary systems:

1. **A modern LLM/VLM reasoner** that behaves like a game-local scientist: it proposes goals, rules, experiments, and high-level plans.
2. **A metacontroller skill system** that compresses repeated behavior into reusable latent options with learned termination conditions.
3. **A world-model and verifier layer** that predicts consequences, checks hypotheses against replay, and plans over learned options rather than primitive actions.

The initial objective is not to train a monolithic agent from scratch. It is to obtain a strong, inspectable system quickly by reusing the existing metacontroller implementation, the official ARC-AGI toolkit and agent scaffold, and mature LLM APIs.

## Architectural principle

The agent should run at three timescales.

```text
SLOW: LLM scientist / strategist
  - infer goals and mechanics
  - propose discriminating experiments
  - update explicit game notebook
  - create or revise high-level plans

MEDIUM: option planner + world model
  - predict option outcomes and duration
  - score plans by progress, information value, cost, and risk
  - verify symbolic and neural predictions against replay

FAST: metacontroller executor
  - select or receive latent skill z
  - steer the frozen behavioral model through the residual stream
  - emit primitive ARC actions
  - continue until the learned switch terminates the option
```

The LLM should not be called every primitive step. It should be called at meaningful boundaries: level start, option termination, prediction failure, novel state, or retrospective reflection.

## Existing code we should reuse

### From this repository

The fork already contains most of the difficult temporal-abstraction machinery:

- `metacontroller/metacontroller.py`
  - split lower/upper autoregressive transformer around the controlled residual stream;
  - behavioral cloning and frozen-backbone discovery paths;
  - acausal discovery encoder and causal internal-RL proposer;
  - learned switching gate and low-rank hypernetwork controller;
  - GRPO/PPO-style and TPO policy losses;
  - next-latent dynamics prediction;
  - external conditioning support through `dim_condition`;
  - evolutionary-strategy support.
- `metacontroller/metacontroller_with_binary_mapper.py`
  - discrete/binary latent codes that may be easier to expose as stable skill IDs to an LLM or symbolic planner.
- `metacontroller/transformer_for_symbolic_grid.py`
  - a spatial front end for symbolic grids.
- Existing BabyAI and PinPad scripts
  - replay-buffer generation;
  - behavior cloning;
  - linear probing;
  - metacontroller discovery;
  - visualization of switch boundaries;
  - internal-RL and evolutionary experiments.

We should adapt these paths rather than create a separate training stack.

### External code and interfaces

Prefer direct reuse over reimplementation:

- **`arcprize/ARC-AGI` / `arc-agi` package**
  - official `Arcade`, `EnvironmentWrapper`, `FrameDataRaw`, `GameAction`, local/offline execution, recordings, and competition-mode behavior.
- **`arcprize/ARC-AGI-3-Agents`**
  - agent loop, action accounting, playback/recording conventions, typed frame structures, and optional tracing.
- **Pydantic**
  - structured notebook, hypotheses, skill cards, plans, and provider-independent LLM outputs.
- **Existing model-provider SDKs**
  - accessed behind a small local protocol so the project does not couple its core logic to one provider.
- **Programmatic-world-model research code when available**
  - borrow verifier and replay-testing patterns; do not duplicate a full coding-agent framework inside the neural package.

The core `metacontroller-pytorch` install should remain lightweight. ARC and LLM functionality should live behind optional dependency groups.

## Target repository layout

```text
metacontroller/
  arcagi3/
    adapter.py              # ARC frames/actions <-> model tensors
    trajectory.py           # canonical transition and episode schemas
    events.py               # frame differencing and event extraction
    notebook.py             # rules, goals, evidence, open questions
    reasoner.py             # provider-neutral LLM/VLM protocol
    planner.py              # primitive and option-level planning
    world_model.py          # neural option-effect model
    program_model.py        # optional executable hypothesis interface
    verifier.py             # replay and prediction verification
    skills.py               # skill registry, prototypes, cards, validation
    agent.py                # three-speed hybrid control loop
    evaluation.py           # action efficiency and ablation metrics

scripts/arcagi3/
  collect.py
  train_bc.py
  probe.py
  discover.py
  build_skills.py
  train_option_model.py
  train_internal_rl.py
  play.py
  evaluate.py

configs/arcagi3/
  smoke.yaml
  symbolic.yaml
  rgb.yaml
  hybrid_llm.yaml

tests/arcagi3/
```

## Canonical data model

All components should communicate through a stable trajectory representation rather than directly through provider- or environment-specific objects.

A transition should include:

- raw layered frame;
- canonical image or symbolic tensor;
- available actions and action arguments;
- selected primitive action;
- game and level state;
- frame-difference events;
- option/skill ID and latent code, when present;
- switch probability and hard boundary;
- LLM notebook revision and reasoning metadata, when present;
- terminal, level-complete, game-over, and reset markers.

Recordings must be deterministic to replay and versioned so later schema changes do not invalidate old datasets.

## Explicit game notebook

The reasoner should maintain a compact, testable notebook rather than an ever-growing prose transcript.

```text
Objects
  id, visual signature, position, state, possible role

Rule hypotheses
  statement, confidence, evidence, counterexamples,
  predicted consequences, discriminating experiment

Goal hypotheses
  candidate terminal condition, confidence, evidence

Skills
  skill ID, empirical description, preconditions,
  expected effects, termination, duration, reliability

Open questions
  uncertainty, competing hypotheses, useful experiment

Current plan
  intended skill sequence, checkpoints, abort conditions
```

LLM output is advisory. Facts enter the notebook only through observed transitions or verifier-backed replay.

## Skill bridge between neural latents and reasoning

The LLM must never be asked to invent or manipulate arbitrary continuous latent vectors directly.

For each segment between learned switches:

1. Store the latent `z`, start state, end state, duration, and salient events.
2. Cluster latents using both code distance and empirical effect similarity.
3. Select a prototype or discrete code for each candidate skill.
4. Intervene from varied starting states to measure reliability and preconditions.
5. Build a `SkillCard` from measured behavior.
6. Let the LLM describe the behavior, while preserving empirical statistics as the source of truth.

Both latent representations should be retained initially:

- continuous Gaussian latents for expressive control;
- binary/discrete codes for stable planner-visible skill identifiers.

## World models

Use two complementary world models.

### Primitive transition model

Predict immediate state/event changes from `(state, primitive action)`. This is useful before meaningful options exist and for verifying local mechanics.

### Option-effect model

Predict from `(state at switch boundary, skill)`:

- next boundary state or state embedding;
- duration in primitive actions;
- termination type;
- goal-progress events;
- failure probability;
- epistemic uncertainty.

Planning over a small number of options should be the default once validated skills exist.

An executable/programmatic model is optional, not mandatory. The LLM may create one when it expects the model to pay for itself. Replay verification should compare its predictions with all relevant recorded transitions. A neural residual model can cover uncertain visual or hidden-state effects that the symbolic hypothesis does not explain.

## Implementation phases

### Phase 0 — Preserve and characterize the upstream baseline

Deliverables:

- keep the fork synchronized with upstream;
- add a small CI matrix and deterministic smoke tests;
- document known-good BabyAI and PinPad commands;
- save a baseline manifest of package versions and seeds;
- verify behavior cloning, discovery, probing, and internal-RL APIs independently.

Exit criteria:

- upstream tests pass;
- a tiny synthetic trajectory completes BC and discovery forward/backward passes;
- no ARC-specific dependency is required for the core package.

### Phase 1 — ARC-AGI-3 adapter and replay pipeline

Deliverables:

- optional `arcagi3` dependency group using the official toolkit;
- `ArcAdapter` for `FrameDataRaw`, layered frames, available actions, reset states, and action arguments;
- canonical trajectory schema and JSONL/replay ingestion;
- frame/event differencer that distinguishes actionable states from animation-only frames;
- random, replay, and scripted smoke agents;
- offline local-game CLI.

Reuse:

- official `EnvironmentWrapper` and `GameAction` interfaces;
- official agent-loop and recorder conventions;
- existing memmap replay-buffer utilities in this repository.

Exit criteria:

- recorded ARC trajectories round-trip without changing actions or frames;
- the same adapter works in local and remote modes;
- unit tests cover action masking, level transitions, resets, and animation frames.

### Phase 2 — LLM scientist baseline

Deliverables:

- provider-neutral `Reasoner` protocol;
- Pydantic schemas for notebook updates, experiments, plans, and actions;
- visual/symbolic observation summarizer;
- evidence ledger and contradiction handling;
- a primitive-action LLM baseline with selective invocation and replay logging;
- deterministic mock reasoner for tests.

The first reasoner should operate outside the metacontroller. This isolates whether the ARC adapter, memory, and experimental loop work before neural skill discovery is introduced.

Exit criteria:

- the reasoner can maintain game-local rules across levels;
- invalid or hallucinated rules are marked unverified and cannot bypass the verifier;
- the baseline can replay its own decision trace.

### Phase 3 — ARC behavioral model and demonstrations

Deliverables:

- convert ARC recordings into the existing replay-buffer format;
- start with symbolic/categorical frames where available, then add RGB/layered-frame support;
- adapt existing BabyAI BC training rather than writing a new trainer;
- support demonstrations from humans, scripted agents, LLM/VLM agents, and successful replay filtering;
- add linear probes for level phase, candidate goal, object role, and action consequence.

Training curriculum:

- procedural hidden-rule games created with the official toolkit;
- public ARC games for integration testing, not the sole source of training;
- failed exploratory trajectories retained for world-model training;
- cleaner successful trajectories prioritized for metacontroller discovery.

Exit criteria:

- BC clearly exceeds random action selection on held-out procedural games;
- hidden phase/goal variables become decodable above simple observation baselines;
- the frozen backbone can be causally steered by supervised diagnostic controllers.

### Phase 4 — Unsupervised temporal abstraction discovery

Deliverables:

- run the existing discovery path on ARC trajectories;
- preserve the frozen backbone requirement;
- compare continuous and binary metacontrollers;
- visualize switches against observed event boundaries;
- add automatic boundary metrics using level events, interactions, inventory changes, and movement phases;
- add fixed-switch, no-switch, co-training, and random-code ablations.

Exit criteria:

- switches are sparse and repeatable across seeds;
- controller interventions cause coherent multi-step behavior;
- discovered codes transfer to changed layouts or later levels more often than random latents;
- the full model beats the forced-every-step-switch ablation.

### Phase 5 — Skill registry and LLM-visible skill cards

Deliverables:

- segment extractor using learned switch boundaries;
- effect-aware latent clustering;
- intervention harness that tests a latent from multiple initial states;
- versioned `SkillRegistry` and `SkillCard` schemas;
- stable skill IDs mapped to continuous prototypes or binary codes;
- reasoner tools to inspect, request, compose, and abandon skills.

Exit criteria:

- every promoted skill has empirical reliability and duration estimates;
- text descriptions are generated from measured examples;
- the LLM can plan using skill IDs without access to raw latent values;
- executing a skill consumes fewer LLM calls than primitive control.

### Phase 6 — Option-level world model and planning

Deliverables:

- option-effect dataset from switch-boundary transitions;
- uncertainty-aware option model;
- beam search or MCTS over skills;
- scoring for goal progress, information gain, action cost, and irreversible risk;
- optional executable hypothesis model with replay verifier;
- planner policy for when to build, repair, consult, or bypass an executable model.

Reuse the existing next-latent dynamics model as an initialization or representation source, but train a separate boundary-to-boundary head for duration and event prediction.

Exit criteria:

- predicted option outcomes calibrate on held-out episodes;
- planning over options improves action efficiency over greedy skill selection;
- verifier failures produce targeted model revisions rather than complete notebook resets.

### Phase 7 — Distillation and internal RL

Deliverables:

- collect `(state, notebook summary, chosen skill)` traces from the LLM-guided agent;
- train the causal action proposer to imitate skill selections;
- apply the repository's GRPO/TPO loss only at switch boundaries;
- use actual task completion and action efficiency as reward;
- retain LLM fallback for high uncertainty or notebook contradictions;
- compare GRPO, TPO, and evolutionary strategies.

Exit criteria:

- the distilled policy handles familiar states without LLM calls;
- internal RL improves either completion or action efficiency over imitation alone;
- policy updates do not degrade validated skill execution.

### Phase 8 — Online reflection and cross-level adaptation

Deliverables:

- post-level retrospective segmentation and notebook revision;
- game-local skill promotion, splitting, merging, and retirement;
- model repair triggered by prediction residuals;
- compact notebook conditioning vector passed through the existing `dim_condition` path;
- context dropout and stale-memory tests.

Exit criteria:

- experience on early levels measurably reduces actions on later levels;
- local adaptation does not alter the frozen global backbone during evaluation;
- stale or false notebook entries can be corrected from evidence.

### Phase 9 — Evaluation and competition hardening

Primary comparisons:

1. random/scripted baseline;
2. primitive-action LLM;
3. LLM + notebook;
4. metacontroller without explicit reasoning;
5. LLM + fixed macro library;
6. LLM + discovered skills;
7. full hybrid with option world model;
8. distilled/internal-RL hybrid.

Metrics:

- games and levels completed;
- human-relative action efficiency;
- primitive actions per level;
- LLM/VLM calls and tokens;
- option success and duration calibration;
- world-model transition/event accuracy;
- notebook rule precision and correction latency;
- transfer benefit from earlier to later levels;
- wall-clock and compute cost.

Evaluation discipline:

- distinguish public-set development from held-out generalization;
- save exact seeds, game versions, model names, prompts, and package versions;
- run ablations under matched action and inference budgets;
- preserve fresh-agent evaluation with no playthrough-specific leakage.

## MVP vertical slice

The first end-to-end milestone should be deliberately narrow:

1. Run one local procedural multi-level ARC-like game through the official toolkit.
2. Record an LLM or scripted demonstration.
3. Train the existing behavioral transformer on those trajectories.
4. Run metacontroller discovery and visualize switch boundaries.
5. Promote at least two validated latent skills.
6. Let the LLM select those skills from a structured notebook.
7. Execute the skills until learned termination.
8. Measure whether later levels require fewer primitive actions and fewer LLM calls than the primitive-action LLM baseline.

This vertical slice tests the central hypothesis without first building a universal world-model system.

## First implementation batch after this roadmap

The next PR should contain only foundational, low-risk work:

- add optional dependency groups for ARC integration and LLM integration;
- create `metacontroller.arcagi3` with trajectory schemas and adapter interfaces;
- implement official-toolkit frame/action conversion;
- ingest and replay official JSONL recordings;
- add a deterministic mock environment and mock reasoner;
- add CLI commands for recording inspection and adapter smoke tests;
- add unit tests and CI coverage.

No LLM vendor SDK, metacontroller training change, or game-specific rule should be required in that first batch.

## Major risks and mitigations

### The LLM overfits familiar semantics

Mitigation: treat priors as hypothesis ordering only; require transition evidence and record counterexamples.

### Discovered latents are not stable skills

Mitigation: promote only after multi-state intervention testing; compare binary and continuous representations; retain supervised diagnostic controllers.

### The neural model learns animation rather than mechanics

Mitigation: identify actionable frames, derive event targets, and train boundary/event predictions in addition to pixels.

### Joint training destroys useful abstractions

Mitigation: preserve the paper's staged procedure and frozen backbone; isolate optional adapters from backbone weights.

### LLM calls dominate action cost or latency

Mitigation: invoke only at boundaries or surprises; cache notebook summaries; distill selections into the causal proposer.

### Public ARC games create false confidence

Mitigation: use procedural hidden-rule curricula and fresh held-out games; report public and held-out results separately.

### The architecture grows before the core hypothesis is tested

Mitigation: prioritize the MVP vertical slice and require measurable transfer before adding richer symbolic modeling.

## Guiding decision rule

At each implementation boundary, choose the smallest addition that answers one empirical question. Prefer adapters around existing code to parallel frameworks, retain ablations, and avoid hard-coding any public game mechanic into the agent.
