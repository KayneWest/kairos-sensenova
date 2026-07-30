# Latent Imagination Planning: Adjusted Plan

Status: adjusted research direction as of 2026-06-05.

This document captures the pivot from trying to prove that a retrofitted Kairos/Sensenova world model is directly action-controllable, toward a more defensible and potentially stronger objective: use action-labeled data as training-time causal scaffolding, then perform optimization over an inspectable tree of imagined futures.

## Core Correction

The earlier framing was too narrow:

```text
video context -> likely future
```

The adjusted framing is:

```text
video context -> distribution/tree of plausible futures -> scored/planned future
```

The model does not need to collapse immediately to one likely future. It can imagine multiple futures, score them, and select or refine a future according to a planning objective.

The key distinction is what controls the branches:

```text
Latent imagination:
z_context + latent sample/plan token -> future_i

Action-conditioned simulation:
z_context + action_i -> future_i
```

Classic Dreamer-style control requires the second form. Our adjusted plan focuses on the first form, with action data used to teach which futures are behaviorally meaningful and to learn bridges from desired futures back to executable actions.

## Thesis

Action data can be used as privileged training signal rather than always being required as an inference-time steering input.

During training:

```text
video + action labels + rewards/events -> better latent dynamics, event representations, inverse dynamics, and plan scoring
```

At inference:

```text
video context -> imagined future branches -> score/optimize/select -> infer actions or goals
```

This makes the action labels a source of causal supervision and data selection, not necessarily the only steering mechanism at inference.

## Why This Fits Our Evidence

Our retrofit attempts showed a recurring pattern:

- Kairos/Sensenova latents support useful future-latent prediction.
- Direct action-conditioned rollout is difficult to recover after visual pretraining.
- Wrong-action controls often remain close to true-action rollouts.
- Closed-loop objectives and action-identifiable filtering help, but the result is not uniformly Dreamer4-equivalent.
- The Dreamer-VLA negative result is consistent with this: video/world-model objectives can learn visual dynamics without surfacing action-causal features strongly enough for control.

The adjusted plan avoids overclaiming direct action control while preserving the useful part: the model can still support planning if we optimize over imagined future branches and expose the imagination process.

## Architecture

Use Kairos/Sensenova as the visual substrate:

```text
pixels -> Kairos latent z
Kairos latent z -> pixels
```

Train additional planning components around the frozen or mostly frozen latent space:

```text
z_context
  -> future proposal model
  -> N latent future trajectories
  -> reward/value/risk/constraint scoring
  -> optimizer/selective planner
  -> inverse dynamics or policy executor
```

Concrete modules:

- `future_proposer`: samples or generates multiple future latent trajectories from context.
- `trajectory_scorer`: predicts reward, success, risk, constraint violation, novelty, or task progress.
- `latent_plan_optimizer`: searches over latent plan tokens or sampled futures.
- `inverse_dynamics`: maps current state plus desired future to action sequences.
- `execution_policy`: maps current state plus selected future/goal token to executable actions.
- `imagination_trace_logger`: decodes and records candidate futures, scores, selected branches, and inferred actions.

## Training Objectives

Use action labels where available, but do not force all inference to depend on raw actions.

Primary losses:

```text
future latent prediction:
z_context -> z_future

multi-branch future modeling:
z_context + noise/plan_token -> diverse plausible z_future_i

reward/event prediction:
z_future_i -> reward/event/success/risk

inverse dynamics:
z_t, z_{t+k} -> action sequence

latent plan distillation:
action/history -> latent plan token u_t

latent plan dynamics:
z_context + u_t -> z_future
```

Control and causality losses:

```text
true action vs zero action
true action vs shuffled action
true action vs time-shifted action
true action vs far-shuffled action
closed-loop rollout contrast
```

Regularization:

```text
action dropout / missing-action embedding
visual-prior preservation
trajectory diversity
plan-token bottleneck
KL or prior penalty for plausible plans
```

## Inference Modes

The system should support three inference modes:

```text
No-action imagination:
z_context -> future branches

Raw-action simulation:
z_context + action sequence -> counterfactual future

Latent-skill planning:
z_context + learned plan token -> future branch
```

The paper should not require all modes to be equally strong. The core adjusted claim is around no-action or latent-skill imagination plus optimization, with raw-action simulation treated as an optional stronger capability if gates pass.

## Planning Loop

At inference:

```python
context = encode_video(frames)

candidates = proposal_model.sample_futures(
    context=context,
    num_candidates=N,
    horizon=H,
)

scored = []
for future in candidates:
    score = scorer(context, future, task)
    risk = risk_model(context, future)
    actions = inverse_dynamics(context, future)
    scored.append((future, score, risk, actions))

selected = planner.select_or_refine(scored)
execute(selected.actions)
log_imagination_trace(scored, selected)
```

This is closer to latent planning than pure Dreamer rollout:

```text
Dreamer:
policy action -> world model rollout -> reward -> policy update

Latent imagination planning:
world model proposes futures -> optimizer selects desired future -> inverse model or policy executes
```

## Observability Requirement

The planning state must be inspectable.

For each decision context, log:

```python
imagination_trace = {
    "context_frames": frames,
    "candidate_futures": [
        {
            "decoded_video": preview_i,
            "future_latents": z_future_i,
            "score": score_i,
            "risk": risk_i,
            "constraint_flags": flags_i,
            "estimated_actions": actions_i,
            "rank": rank_i,
        }
    ],
    "selected_candidate": selected_idx,
    "selection_reason": planner_summary,
}
```

Minimum deliverable:

- Decode top `K` imagined futures to video.
- Show score/risk for each branch.
- Show selected branch.
- Show inferred action sequence or plan token.
- Compare selected branch against rejected alternatives.

This becomes the "imagination browser" for paper figures and debugging.

## Data Role

Action data helps in two ways:

1. It identifies parts of the dataset where behaviorally important transitions happen.
2. It provides supervision for inverse dynamics, latent plan tokens, and action-effect representations.

The best data is not merely large video. It is data where wrong actions imply wrong futures.

Preferred data:

- SOAR, especially high-action/high-visual-delta windows.
- LIBERO for clean task labels and benchmarkable manipulation.
- BridgeData V2 for real-robot tabletop behaviors.
- RT-1/Fractal and selected Open X-Embodiment subsets for scale.
- DROID only after filtering, because our audit found weak incremental action signal beyond scene history.

Training should prioritize filtered windows:

```text
high action magnitude
visible/latent scene change
object contact or manipulation event
task progress or success label
nontrivial inverse-dynamics predictability
```

## Model Scale Direction

Keep Kairos/Sensenova frozen initially.

Scale the planning/dynamics layer instead:

```text
frozen Kairos tokenizer/VAE
  +
50M-200M parameter latent dynamics/planning transformer
```

Initial serious target:

```text
d_model: 512
layers: 12-16
heads: 8
context: 16-32 frames
trainable params: 50M-100M
```

Larger target:

```text
d_model: 768
layers: 16
trainable params: 150M-200M
```

The goal is not "non-Kairos". The goal is:

```text
Kairos visual prior + larger trainable latent imagination/planning system
```

## Training Budget Estimate

Our previous large runs were step-heavy but not epoch-heavy:

- Action-focus continuation: about `0.42` effective epochs.
- Closed-loop continuation: about `0.18` effective epochs.
- HF rich-action runs: about `0.66` effective epochs.

For a real reshape attempt:

```text
minimum: 5M action-identifiable windows, 5 effective epochs
better: 10M-20M filtered windows, 5-10 effective epochs
```

At global batch `64`:

```text
25M samples -> about 390k optimizer steps
50M-200M samples -> about 0.8M-3.1M optimizer steps
```

This is why the prior runs should be described as pilot-scale continued training, not full reshaping.

## Evaluation Gates

Do not claim controllable simulation unless action-conditioned gates pass.

For raw-action mode:

```text
true action rollout beats zero action
true action rollout beats shuffled action
true action rollout beats time-shifted action
true action rollout beats far-shuffled action
closed-loop rollout beats persistence
```

For latent planning mode:

```text
top-scored futures predict task progress better than random futures
selected futures decode into coherent plausible videos
inverse dynamics can execute or approximate selected futures
selected plans beat behavior cloning / zero-action / random-plan baselines
imagination traces are stable across seeds
```

For observability:

```text
top-K decoded futures are visually distinct
scores correlate with outcome labels
selected branch is not always the highest-probability scene continuation
rejected branches show interpretable failure modes
```

## Paper Claim Boundary

Unsafe claim:

```text
Kairos/Sensenova is a Dreamer4-equivalent action-conditioned simulator after retrofit.
```

Safe adjusted claim:

```text
Action-labeled data can scaffold a pretrained video world model into an inspectable latent imagination system for planning over future trajectories.
```

Stronger claim if experiments pass:

```text
The system optimizes over decoded imagined futures and improves policy selection by choosing high-scoring latent trajectories, while preserving observability into the planning process.
```

Optional stronger claim only if raw-action gates pass:

```text
The learned latent dynamics also supports action-conditioned counterfactual rollouts under strict zero/shuffle/time-shift/far-shuffle controls.
```

## Immediate Next Experiments

1. Build an imagination trace format.
2. Implement top-K latent future sampling from Kairos context.
3. Decode top-K futures to video previews.
4. Train a trajectory scorer on available reward/event/success labels.
5. Train inverse dynamics from selected future latents to actions.
6. Evaluate whether selected futures beat random futures and behavior-cloning futures.
7. Add a simple CEM-style optimizer over latent plan tokens.
8. Produce an "imagination browser" figure for the paper.

## Success Criteria

Minimum paper-worthy result:

- The model generates multiple plausible futures from the same context.
- A learned scorer ranks futures in a way correlated with task progress or success.
- The selected future differs from naive most-likely continuation.
- The inferred action plan from the selected future beats a baseline policy in learned-simulator evaluation.
- The full imagination trace is observable and decodable.

High-confidence result:

- The above holds across held-out tasks and seeds.
- The selected branches beat random, BC, zero-action, and shuffled-plan controls.
- The imagination browser shows interpretable rejected futures and selected futures.

## Summary

The adjusted plan is not to abandon action data. It is to use action data more strategically:

```text
actions as causal scaffolding during training
latent futures as the planning object at inference
optimization over imagined branches
decoded traces for observability
inverse dynamics or policy for execution
```

This avoids the trap of claiming direct Dreamer4 replication from a retrofit while preserving the central ambition: an agent that thinks by imagining futures, scores them, and exposes its planning process.
