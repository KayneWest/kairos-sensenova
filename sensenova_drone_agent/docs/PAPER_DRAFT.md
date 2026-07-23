# From Scene Priors to Decision-Quality Imagination: Retrofitting Action Grounding into Video World-Model Latent Spaces

*Draft v1.7 — 2026-07-23. Numbers are final from the July 2026 campaign; see
WORKLOG.md for artifact paths. Bracketed notes mark writing TODOs.*

## Abstract

Video world models promise agents that *think in frames*: imagine futures
under candidate actions, evaluate them, act on the best. We study this loop
directly, atop a frozen video tokenizer, with a latent planner that
imagines, scores, and inverts futures into actions. Our first contribution
is methodological: the standard training-time selection metric is
circular — under an external audit (future error vs the real future,
persistence baselines, candidate banks) the scorer was near-perfectly
*anti-correlated* with reality on robot data (r = −0.97) while the training
metric read healthy. Four audit-driven fixes — candidate-ranking
supervision, unit-norm plan tokens, reward-path stop-gradients, and
per-step plan conditioning — close the offline loop: selection beats random
on every trained source and transfers to a never-trained source, while a
reproducible phase transition grounds exact action timing (error ratios
~1.0 → 11–27x). Any act-time head given the plan token
learns to bypass imagination; plan-dropout is the general cure. In a
closed-loop drone game, selecting behavior-prior candidates by imagined
value beats acting without thinking in one seed (n = 1000, CI-clean) and
*reverses, CI-clean, in a second seed with equal-or-better offline metrics*:
offline decision-quality gates do not predict even the sign of act-time
selection value. The gap is distributional, and one DAgger iteration closes
it: after training on 400 episodes of the agent's own experience, thinking
beats both controls in both seeds (success 5.7–6.1% vs 0.1–0.5% without
thinking; positive mean return; all CIs clear of zero). The repair does not
iterate: a second round of self-collected data fails at every mixture
tested, and training on the improved agent's failure-concentrated episodes
re-inverts selection to worse-than-random. A judge/imagination exchange
experiment localizes that poison: the "inverted" stack's value head ranks a
healthy model's futures as well as a good one, while no judge rescues
selection among the poisoned model's futures (8/8 cells, two seeds) —
failure-concentrated self-training corrupts the *dreams*, not the
judgment. A complementary test replaces argmax search with value-guided
diffusion sampling: trained by pure likelihood, the generative proposer
recapitulates scene priors (action-conditioning inert), and guidance over
an action-blind prior collapses into passivity under either judge.
Deliberation needs an action-causal imagination before it needs a better
judge. Finally, replacing outcome labels with *corrective* ones — an
expert's action at each state the agent visits, classic DAgger — ladders
where self-imitation could not: the whole stack clears its previous
ceiling in one round and roughly doubles it before plateauing (~10% vs
41.5% expert), locating the next binding constraint in the frozen
representation rather than the data. A second domain replicates the
whole arc at scale: on ViZDoom, with the *drone game's* tokenizer
(never trained on a Doom frame) and a 100%-success scripted teacher,
one round of clean corrective labels lifts selection to a 78–84%
survival ceiling in both seeds and saturates immediately; a
drift-corrupted variant of the same teacher tops out below half that,
and under data aggregation the behavior-cloned proposer degrades
monotonically while argmax selection over the fixed imagination holds
the ceiling — corrective-label quality and representation capacity
bind, in that order, while selection acts as a robustness layer.

## 1. Introduction

An agent with a video world model should be able to deliberate: given its
recent frames, spawn several candidate futures, judge which future is best,
and emit the actions that realize it. Dreamer-style agents implement a
version of this with policies trained inside imagination; a complementary
line retrofits *frozen* video generative models with action interfaces. The
appeal of the retrofit path is that the visual knowledge is already paid
for; the risk, documented repeatedly, is that such models learn *scene
priors* — context → likely future — rather than action-causal dynamics.

This paper asks the question at the level where it matters: not "can the
model imagine?" but "**does imagining and evaluating candidate futures
improve the action taken?**" We make that question operational, audit a
planner against it, repair what the audit exposes, and take the result to a
closed-loop environment.

Contributions:

1. **A decision-quality audit with external proxies** (§3). Candidate
   futures are judged by their error against the *real* future, against
   persistence baselines, with candidates drawn from the empirical plan
   distribution ("bank" candidates), across K-sweeps and held-out sources
   with bootstrap CIs. The audit exposed that the standard training metric
   (selected-minus-random in the scorer's own score space) is circular: it
   read +0.19 while true selection was worse than random on every robot
   source, with scorer–fidelity correlation −0.97.
2. **Four verified fixes that close the offline loop** (§4): a reward-free
   candidate-ranking loss; unit-norm plan tokens (sampling in-distribution
   by construction); stop-gradients on the reward path (a latent bug that
   activates once the scorer becomes fidelity-sensitive and then drags
   imagined futures away from reality); and per-step plan conditioning. The
   final configuration passes all decision-quality gates on game sources and
   transfers selection to a never-trained source.
3. **A reproducible timing phase transition** (§4.3, Fig. 1): with an
   *absolute* contrast margin, the margin-to-error ratio anneals upward as
   fidelity improves, and once easy negatives saturate, gradient
   concentrates on timing negatives — wrong-timing error ratios move off
   1.0 (reaching 5.3x within ~18k steps of onset in the control arm and
   11–27x at the final checkpoints), in both seeds, with onset tracking the
   transition rather than a step count.
4. **The plan-shortcut symmetry** (§5): any head that receives the plan
   token at act time learns to bypass imagination — the inverse-dynamics
   head decodes actions from the plan (verified: shuffling the future
   changes nothing; removing the plan degrades 4.6x), and the scorer
   becomes Q(state, plan), ignoring an imagined future that already
   *predicts the crash*. Plan-dropout training, applied to each head,
   restores the imagination pathway; a cross-context imagined-inverse loss
   ("only the future can explain the target") closes offline acting.
5. **Closed-loop evaluation with matched-seed paired controls** (§6,
   Fig. 2): nine rounds in a drone navigation game, including a structural
   dataset finding — standard window enumeration can never place an
   episode's terminal transition inside the scored future, so models train
   without ever seeing an outcome; absorbing-state padding fixes it — a
   diagnosis of offline-MPC model exploitation, and a behavior-anchored
   remedy under which thinking beats acting-without-thinking on success and
   return with CIs clear of zero at n = 1000 — followed by a full-stack
   repeat in an independent seed in which every comparison *reverses*,
   CI-clean, under equal-or-better offline metrics. The behavioral claim is
   therefore withdrawn; the seed-level sign instability is the sharper
   finding — and one DAgger iteration on the agent's own episodes restores
   the win *consistently in both seeds* (Fig. 2, right panels), showing the
   failure is distributional rather than intrinsic. Iterating the repair
   does not compound (§6, Fig. 5).
6. **A judge/imagination exchange test that localizes self-training's
   poison** (§6.1, Fig. 6): swapping value heads and proposers between the
   healthy and regressed stacks shows closed-loop success follows the
   imagination in every cell and is invariant to the judge — the
   "inverted" value head is exonerated. A value-guided diffusion proposer
   (§6.2) completes the picture: likelihood-only generative training
   recapitulates the scene-prior failure, and gradient guidance has no
   leverage on an action-blind prior.

Throughout we maintain a strict claim discipline (§9): no claims about real
drones; margins reported with CIs; unresolved comparisons stated as such.

## 2. Setup

**Backbone.** A Dreamer4-style video tokenizer (time-causal encoder,
d_model 128, 16 latents × 32 dims per frame, trained on a mixed corpus) is
frozen throughout; all planning happens in its latent space. Sources: three
dm_control-like game datasets (expert and mixed play), SOAR robot
manipulation, DROID, Fractal, RoboNet, and Bridge — Bridge is excluded from
training entirely and serves as the held-out source.

**Planner.** A plan encoder maps (context encoding, action chunk) to a
128-d plan token u; a GRU proposer rolls imagined future latents from
(context, u); a scorer predicts return/value from (context, imagined
future, u); an inverse-dynamics head maps (context, future, u) to an action
chunk. Contrast losses require wrong-action plans (zero, shuffled,
time-shifted, permuted, reversed) to imagine worse futures. Context length
8, horizon 8, at 128×128 resolution. Fig. 0 shows the perceive/think/act
lanes, the placement of the four fixes (§4), and the plan-free act-time
heads (§5).

**Gates.** For proposing: wrong-over-true error ratios and
true-over-persistence at h ∈ {4, 8, 16}. For evaluating: scorer–fidelity
correlation on bank candidates, oracle rank of the true plan,
selected-vs-random future error with K ∈ {1..64}. For acting: selected
actions vs logged actions against random/zero/mean/repeat-last controls and
oracle ceilings. All with bootstrap CIs, per source.

## 3. The audit, and what it found

The planner under audit had trained 130k steps and reported a healthy
selection metric (candidate_selected_minus_random = +0.19). That metric
compares the argmax-scored candidate with a random candidate *in score
space*: it is positive whenever scores have spread, regardless of whether
scores mean anything. The audit replaced it with external ground truth.

Findings at 130k (8 sources × 256 contexts):

- **The proposer half-works.** True-action futures beat persistence at
  h ≤ 8 on trained sources and beat zero/shuffled actions by 4–12x /
  1.1–1.7x in future error. Exact timing is not grounded: time-shifted
  actions are indistinguishable (ratios 1.00–1.09) everywhere.
- **The scorer is inverted.** On SOAR and held-out Bridge, scorer–fidelity
  correlation on bank candidates is −0.97 and −0.96; the zero-action plan
  *outscores* the true plan on every robot source (margin −3 to −5.5);
  argmax selection picks candidates farther from reality than random (0%
  beats-random).
- **Sampling is out-of-distribution.** Encoded plan norms are 69–137
  versus 11.3 for the N(0,1) sampler used at eval; the scorer partially
  keys on norm.
- **It is not undertraining.** A 50k/90k/130k checkpoint sweep shows the
  inversion is stable.

Root causes, each verified: the scorer was trained only as return
regression on the true plan (never seeing a candidate); return targets are
degenerate on several robot exports (identically zero reward) and
unlearnable on others; and plan geometry was unconstrained.

## 4. Closing the offline loop

We repaired the loop in controlled arms, each audited identically.

### 4.1 Rank the candidates; normalize the plans; detach the reward path

A reward-free **candidate-ranking hinge** trains the scorer on pools of
true/control/bank/Gaussian plans, ranked by each imagined future's error
against the real future (gradients confined to the scorer head).
**Unit-norm plan tokens** make sphere sampling in-distribution by
construction. Ranking alone repaired the scorer (fidelity correlation
−0.97 → +0.64 on SOAR; −0.96 → +0.38 held-out) but exposed a latent bug:
once the scorer is fidelity-sensitive, the *undetached reward-regression
path* drags imagined futures toward score targets — in both arms true-plan
future error collapsed from 0.7–1.5x persistence to 8–18x, with train
future loss rising 5–10x. **Stop-gradients on all scorer inputs in the
reward loss** (Dreamer-style head-only training) eliminated the collapse.

A margin ablation produced a clean dichotomy: a *relative* contrast margin
(release at 1.2x) preserves fidelity by collapsing plan-sensitivity —
accurate but action-independent imagination, useless for thinking — while
an *absolute* margin keeps pressure on and, as fidelity improves, demands
ever-larger relative separation.

### 4.2 Per-step plan conditioning

Raw-action analysis showed timing is identifiable in the data for game
sources (shift-1 action cosine 0.33) yet never learned — an architectural
gap: nothing tied rollout step t to the action at step t. A per-step
readout (plan token → horizon per-step embeddings) closes it while keeping
the plan a single samplable token.

### 4.3 The timing phase transition (Fig. 1)

Both the objective-only arm and the per-step arm undergo a phase
transition in which all timing contrasts (shift, permutation, reversal)
rise together from ~1.0 to 5–22x while future error is unchanged. The
mechanism is the absolute margin's implicit annealing; the per-step arm
transitions ~7k steps earlier and reaches better fidelity at matched steps.
The transition reproduces in an independent seed, with onset shifted — the
scorer's transient robot-source inversion tracks the transition, not the
step count, and *recovers* (Fig. 3), motivating per-seed, metric-based
checkpoint selection.

**Final offline results** (Table 1): the seed-1 full sweep at 210k passes
6/6 gates on all three game sources (timing 11–15x, zero 10–13x, selection
beats random 93–94%, persistence 0.30–0.41); the headline seed-2 final
(210k) is the best overall artifact — expert 6/6 (timing 27x), SOAR 5/6
(timing grounded at 5.0, selection beats random 87%), and held-out Bridge
5/6 with selection beating random in 100% of contexts. The single failing
robot gate at the headline checkpoint is the true-vs-shuffle *score margin*.

**Table 1a — decision-quality gates at the 130k baseline** (256
contexts/source; mse ratios > 1 good; fid_corr on bank candidates;
sel−rand @ K=64 in future-MSE, < 0 good). Persistence at 130k: trained
robot sources 0.52–0.94 at h4/h8 with h16 collapse (3.4–5.8x); held-out
Bridge worse than persistence at all horizons (3.2 / 5.6 / 19x). No source
passes all gates.

| source | zero | shuffle | tshift | fid_corr | sel−rand @K64 | verdict |
|---|---|---|---|---|---|---|
| dreamer4_hf_expert | 4.33 | 1.23 | 1.03 | +0.02 | +0.0010 | scorer flat |
| dreamer4_hf_mixed_small | 5.63 | 1.14 | 1.02 | −0.08 | +0.0026 | scorer harmful |
| dreamer4_hf_mixed_large | 6.28 | 1.22 | 1.00 | −0.17 | +0.0029 | scorer harmful |
| soar_native_v2 | 10.19 | 1.71 | 1.01 | −0.97 | +0.0328 | scorer inverted |
| hf_robot_droid | 11.65 | 1.00 | 1.00 | +0.15 | +0.0004 | actions ignored beyond zero |
| hf_robot_fractal | 1.91 | 1.01 | 1.01 | −0.40 | +0.0147 | scorer inverted |
| hf_robot_bridge (held out) | 2.59 | 1.74 | 1.04 | −0.96 | +0.0114 | scorer inverted |
| robonet_sample_64 | 1.06 | 1.05 | 1.01 | −0.27 | −0.0040 | spurious pass (tiny source) |

**Table 1b — final rankfix checkpoints** (same battery; persist = true-plan
future error over persistence, < 1 good; all cells re-extracted from the
per-run `output/decision_quality_audit_*/summary.json` records).

| source | checkpoint | fid_corr | zero | shuffle | tshift | persist h4/h8 | persist h16 | beats-rand @K64 | gates |
|---|---|---|---|---|---|---|---|---|---|
| expert | armE 150k (seed 1) | +0.50 | 11.95 | 7.97 | 2.92 | 0.43 / 0.40 | 2.48 | 91% | 6/6 |
| SOAR | armE 150k (seed 1) | +0.59 | 22.96 | 1.52 | 1.02 | 0.49 / 0.63 | 5.42 | 88% | 5/6 |
| Bridge (held out) | armE 150k (seed 1) | +0.89 | 18.40 | 2.48 | 1.12 | 1.16 / 1.60 | 5.45 | 99% | 5/6 |
| expert | armE 210k (seed 1, full sweep) | +0.57 | 9.59 | 12.26 | 11.49 | 0.36 / 0.36 | 2.21 | 93% | 6/6 |
| mixed_small | armE 210k (seed 1, full sweep) | +0.66 | 11.02 | 15.63 | 14.00 | 0.37 / 0.41 | 2.06 | 94% | 6/6 |
| mixed_large | armE 210k (seed 1, full sweep) | +0.63 | 12.71 | 17.72 | 15.05 | 0.32 / 0.30 | 2.09 | 94% | 6/6 |
| expert | armD 210k (control) | +0.43 | 17.45 | 24.29 | 22.10 | 0.28 / 0.40 | 2.81 | 96% | 6/6 |
| SOAR | armD 210k (control) | +0.48 | 37.29 | 1.34 | 3.59 | 0.18 / 0.21 | 1.69 | 77% | 4/6 |
| Bridge (held out) | armD 210k (control) | +0.66 | 28.99 | 1.61 | 1.88 | 0.21 / 0.36 | 2.27 | 84% | 5/6 |
| expert | armE_seed2 210k (headline) | +0.51 | 26.56 | 29.04 | 27.01 | 0.21 / 0.20 | 2.28 | 87% | 6/6 |
| SOAR | armE_seed2 210k (headline) | +0.33 | 43.75 | 2.95 | 4.99 | 0.32 / 0.51 | 4.36 | 87% | 5/6 |
| Bridge (held out) | armE_seed2 210k (headline) | +0.66 | 31.79 | 4.04 | 2.36 | 0.75 / 1.20 | 11.81 | 100% | 5/6 |

## 5. Acting on thoughts: the plan-shortcut symmetry

Decoding selected futures into actions initially failed in a specific way:
with the true plan, the inverse head is near-perfect *regardless of the
future* (shuffling the future: 0.0153 vs 0.0150; removing the plan: 4.6x
worse). The head decodes the plan — which encodes the true actions during
training — not the future. Borrowed candidate plans therefore emit the
wrong context's actions no matter how good the selected future is.

The cure is **plan-dropout** plus supervision of the exact act-time path:
an imagined-future inverse loss (decode true actions from the imagined
future with the plan zeroed) and a **cross-context** variant — encode
*another* context's action chunk, imagine the future it implies here, and
require the head to recover those foreign actions from the future alone.
Context cannot explain the target; only the future can. With these, offline
acting passes 5/5 gates on game data (selected actions beat random
candidates with CI, beat the blind mean-action prior, and improve
monotonically with K) and 4/5 on SOAR. Margins are thin (~1.5% of action
error on game data, ~0.7% on SOAR) relative to the oracle ceiling — the
128-d plan bottleneck — and Bridge transfer remains open (1/5 gates).

The same disease reappeared in the closed-loop scorer (§6): given the plan,
it became Q(state, plan) and ignored an imagined future that correctly
predicted a crash. Score-plan-dropout — the same fix — produced the first
behavioral improvement. We state this as a design rule: **no head consumed
at act time should receive the plan token without dropout training.**

## 6. Closed-loop: the drone navigation game

We instantiated the full stack in a first-person drone navigation game
(discrete 9-action control, procedurally placed obstacles, shaped rewards
with success/collision terminals; a scripted expert achieves 41.5%
success). In-domain offline metrics are near-ceiling (value correlation
+0.93, scorer fidelity +0.94, zero-action ratios > 100x). Closed-loop
evaluation uses matched episode seeds across policies and paired bootstrap
CIs.

**Table 2 — closed-loop rounds** (each removes one verified defect;
"heuristic ceiling" = scripted expert at 41.5% success).

| Round | Defect found | Fix | Behavioral outcome |
|---|---|---|---|
| 1 | Plan-free discrete-argmax decoding collapses to 3 of 9 actions (forward 65%, yaw never; 45% accuracy even on real futures) — the mean-action trap in discrete form | Action-native MPC: candidates are action chunks encoded via the plan encoder; the winner's own actions execute (no inverse head) | All imagination policies crash in ~9 steps |
| 2 | Scorer myopia: 8-step shaped return rewards charging at the goal; the collision lies beyond the credit window | Relabel rewards as full-episode discounted return-to-go (gamma 0.99) | Crashes even faster (5.9 steps) while "winning" short-horizon return |
| 3 | Return-to-go value labels alone change nothing | Retrain scorer as episode-value head (gamma 0) | Behavior identical to round 2 — exposes round 4 |
| 4 | Structural data bug: window enumeration can never place an episode's terminal transition inside the scored future; the planner never trained on a future containing a collision (pre-collision contexts ~0.6% of windows) | Absorbing-state padding in the collector (repeated final frame, hover action, zero reward) | Behavior still identical — exposes round 5 |
| 5 | Scorer plan-shortcut (the §5 disease): scorer routes through the plan token, Q(ctx, plan), scoring forward +16.8 while the imagined future correctly predicts the crash | Score-plan-dropout 0.5 + act-time plan-free scoring | First behavioral movement: survival 5.9 → 14.0 steps, first success (0.5%), paired return +4.0 [+3.3, +4.7] |
| 6 | Hypothesis: missing counterfactual data at decision points | True counterfactual branch data via env snapshot/restore — all 9 actions from the same state (4,221 branches at blocked-front states) | Offline objectives improve; closed-loop regresses (6-step crashes, negative return delta vs random selection) |
| 7 | Act-time argmax searches outside behavior support (offline-MPC model exploitation) | BC-anchored thinking: BC chunk head proposes K = 32 chunks; imagination + plan-free value select among them | First behavioral positive: think 3.5% success / −1.04 return vs BC 0.0% / −3.53; think−BC success +3.5pp [+1.0, +6.5], return +2.49 [+1.57, +3.63] |
| 8 | Weak BC anchor (60% action accuracy); statistical power | Stronger anchor (plateaus at 60.9% — encoder-capped, not data-capped) + powered eval, n = 1000 matched seeds | CI-clean within seed: think 2.8% / −0.42 vs BC 1.4% / −2.16; success +1.4pp [+0.4, +2.6], return +1.74 [+1.20, +2.25]; return vs random +1.78 (25.7 vs 37.2 steps) |
| 9b | Seed-robustness untested | Full-stack repeat with an independent training seed (fresh eval seeds, n = 1000) | Sign reversal, CI-clean both sides: think 0.9% vs BC 2.2% success; think−BC success −1.3pp [−2.3, −0.3], return −2.26 [−2.73, −1.83] — despite equal-or-better offline metrics |
| PMPO | Act-time argmax over a seed-fragile value head is the residual failure | Offline PMPO policy trained in imagination (K = 16 sampled chunks, sign-of-advantage weighting, KL to a built-in BC prior; two seeds, n = 1000) | Learns survival (72–76 steps, few collisions) but ~0% success; return-vs-BC flips sign across seeds; thinking adds nothing over the policy — no cross-seed-consistent win |

Nine rounds, each removing one verified defect (Table 2):
inverse-head discrete-argmax collapse (the mean-action trap in argmax
form); scorer myopia under 8-step shaped returns; return-to-go value
labels; a **structural dataset finding** — window enumeration requires
windows to fit inside episodes, so a terminal transition can *never* occur
inside the scored region: the model had never trained on any future
containing a collision (absorbing-state padding fixes this); the scorer
plan-shortcut (§5), whose repair produced the first behavioral movement
(2.4x survival, paired return CI > 0); and true counterfactual branch data
collected via environment snapshot/restore — same state, all nine actions
rolled out — which improved offline objectives but *worsened* closed-loop
behavior.

That last observation crystallizes the diagnosis: **offline decision-quality
gates do not certify act-time search.** Argmax over imagined futures
reliably finds states where the model is confidently wrong and drives into
them; each fix moved the offline numbers while behavior stagnated. This is
the classic offline model-exploitation failure and the reason Dreamer
trains a policy inside imagination rather than searching at act time.

**Behavior-anchored thinking, and the two-seed test.** Restricting
candidates to a behavior prior's support — a BC chunk head proposes K = 32
plausible action chunks; imagination and the plan-free value scorer select
among them — produced, in the first training seed, a CI-clean behavioral
positive (Fig. 2, first panel): 2.8% success and −0.42 return for think-then-act versus
1.4% / −2.16 for the same BC acting without thinking (success +1.4pp,
CI [+0.4, +2.6]; return +1.74, CI [+1.20, +2.25]), with directed
goal-reaching versus random selection (+1.78 return; 25.7 vs 37.2 steps).
A full-stack repeat with an independent training seed **reversed every
comparison with CIs clear of zero** (think 0.9% vs BC 2.2% success; return
−2.26 vs BC) — despite the repeat seed's *offline* metrics being equal or
better (scorer fidelity +0.78 vs +0.57, return correlation +0.78 vs +0.76,
near-identical BC accuracy 60.7% vs 60.9%). We
therefore do not claim a behavioral improvement. What the two-seed test
establishes instead is stronger than the positive would have been: offline
decision-quality metrics, even in-domain and near ceiling, do not predict
the **sign** of act-time selection value; the value head's act-time
preferences are seed-fragile in ways invisible to every offline gate. This
is the sharpest form of our central methodological finding and the direct
motivation for policy-in-imagination over act-time value search (§10).

**One DAgger iteration closes the gap — consistently.** If seed-fragility
is distributional, training on the agent's own visited states should remove
it. We collected 400 episodes with the think-then-act agent itself (387
ended in collision — its failures became supervision), relabeled with
return-to-go, retrained the planner and BC head on the mixed data, and
re-evaluated at n = 1000 — then repeated the entire training with an
independent seed. Both seeds pass the strict gate (Fig. 2, right two
panels): success 6.1% / 5.7%
versus 0.5% / 0.1% for the same behavior prior acting without thinking
(delta +5.6pp in both, CIs clear of zero) and 1.4% / 2.2% for random
selection among identical candidates (+4.7pp / +3.5pp, CI-clean); mean
return turns positive in both seeds (+1.54 / +1.60) against negative for
every control. The pre-DAgger reversal and post-DAgger consistency together
give the paper's constructive conclusion: act-time value selection fails
not intrinsically but *distributionally*, and a single round of on-policy
data collection is sufficient, in this domain, to convert imagination into
a reliable behavioral advantage.

**Iterating the loop: a repair, not a ladder (Fig. 5).** A second round —
collect 400 episodes with the improved cycle-1 agent (24 successes versus
cycle 1's 13; longer, more purposeful episodes; still 94% collisions),
retrain, and re-evaluate at n = 1000 with two independent training seeds
per configuration — fails to reproduce the win under *every* data recipe
tested. Accumulating the new data (DAgger fraction ½): 0.3% / 2.2% success,
seed 1 CI-clean below both controls. Rebalancing to cycle 1's ⅓ fraction:
partial recovery in one seed (2.9%, beating the prior CI-clean but tying
random selection) and null in the other (0.7%). Replacing the old
self-data entirely (base + cycle-2 data only — the exact cycle-1 recipe
shape with fresher data): the strongest failure, in both seeds — 0/1000
and 5/1000 successes, with selection *worse than random* CI-clean (seed 1:
return −6.18, success −1.7pp; seed 2: success −1.4pp [−2.4, −0.5]): the
scorer inversion of §3 re-emerges, produced this time by the data rather
than the objective. The mechanism is visible in the data's shape: the
improved agent's episodes are long, goal-directed flights that mostly end
in collision, so the training signal associates goal-directed prefixes
with crash-shaped continuations. As the agent improves, its failures
concentrate along its best trajectories, and outcome-labeled
self-experience becomes negative evidence against goal-directed action —
and §6.1 shows by direct exchange that this poison settles in the learned
*dynamics* (the imagination), not the value head, our own first reading
notwithstanding. Classic DAgger escapes
this by querying an expert on the visited states; self-imitation has no
such corrective signal and no reason to improve monotonically. The
constructive conclusion above therefore carries a sharp boundary:
on-policy data closes the distribution gap *once*; it is not, by itself, a
self-improvement ladder.

### 6.1 Where the failure lives: exchanging judges and imaginations (Fig. 6)

The regressed cycle-2c stack selects *worse than random* — but a stack has
three suspects: its candidate prior, its imagination (proposer), and its
judge (value head). Because all stacks share one frozen tokenizer latent
space, components are exchangeable at act time: we run BC-anchored
argmax thinking with the proposer from one stack and the judge (its own
context encoder + scorer, plan-free) from another, at n = 1000 with
matched seeds, two independent training seeds per cell. The result is
unanimous (success, seed 1 / seed 2; strict gate):

| | honest judge (cycle 1) | "inverted" judge (cycle 2c) |
|---|---|---|
| healthy imagination (cycle 1) | 5.3% / 6.2% — pass, pass | 6.2% / 4.5% — pass, pass |
| poisoned imagination (cycle 2c) | 0.0% / 0.5% — fail, fail | 0.0% / 0.7% — fail, fail |

Success follows the imagination in every cell and is invariant to the
judge. The cycle-2c value head, certified "inverted" inside its own stack,
ranks a healthy model's imagined futures as well as the good value head
does; the good value head is powerless among the poisoned model's futures.
And since worse-than-random selection requires *anti-correlation*, the
poisoned proposer is systematically anti-goal: trained on episodes where
goal-directed flight ends in collision, it imagines crash-shaped futures
precisely for good action chunks, and an honest judge shown corrupted
dreams faithfully selects against the goal. This retrospectively re-aims
the amplifier finding: across every configuration we can decompose,
deliberation amplifies the quality of the *imagination* (think-success
spans 0.0–6.2% with the proposer, holding the judge fixed either way).
Applying the same exchange to the §6 pre-DAgger reversal stacks answers
the natural follow-up: there, the effect is *distributed* — each seed-2
component (imagination and judge) degrades thinking mildly on its own and
the pairing compounds (think-success 2.5% / 1.6% / 2.1% / 0.4% across the
four cells at n = 1000) — so imagination-dominance is the signature of
*data poisoning* specifically, while generic seed-fragility in the
low-signal regime is spread across the whole imagination-judge system.

### 6.2 Value-guided generation: soft thinking needs an action-causal prior

If hard argmax amplifies whatever the imagination-judge system believes, a
natural remedy is *soft* selection: sample one future from a generative
prior and steer the sampling trajectory with the value gradient
(classifier-guided diffusion — "force the latent down a thinking
trajectory before releasing it"), so the prior continuously regularizes
the optimization. We trained a conditional latent-diffusion proposer over
future chunks (persistence-delta parameterization, plan-dropout
conditioning from the frozen planner encoders; sample error at or below
persistence) and evaluated closed-loop under both judges, two seeds,
n = 1000, with a guidance-scale sweep.

Two negative results, both two-seed-consistent. First, *likelihood is not
enough*: the diffusion proposer reaches persistence-level sample quality
while plan-conditioning stays inert (conditioned vs plan-free sample error
ratio ~1.0 throughout training) — the scene-prior pathology of §1,
recapitulated in a second model class; candidate-argmax over its samples
is judge-noise selection, indistinguishable from random (both judges, both
seeds). Second, *guidance cannot rescue an action-blind prior*: guided
sampling collapses into passivity (0% success; 43–80% timeouts; the
plan-free inverse decodes near-persistence futures as hover — the
absorbing-state pairing), and the guidance scale is flat over two orders
of magnitude. The floor property that motivated the design does hold —
under the inverted judge, guided selection sits at the prior's level
rather than below random — but it holds vacuously, because guidance has no
leverage anywhere. Two follow-ups complete the picture. *Soft search on the plan manifold*
(gradient ascent on the judge's score through the action-conditioned GRU
proposer, re-projected onto the unit-norm plan sphere; actions decoded
plan-free from the optimized future) is no rescue either: in one seed it
beats no-thinking BC (CI-clean) but ties random selection at 1.5%
success; in the second it sits at BC's floor (0.1–0.2%) — judge-invariant
in both, far below hard argmax's 4.5–6.2%, and seed-fragile even in its
failure mode. The plausible mechanism is the §5 thin link: argmax
executes a discrete, guaranteed-valid behavior-prior candidate, while any
soft method must decode actions from an optimized future through the
inverse head. And *making the generative prior action-causal by
transplanting the contrast recipe* proves to be a knife-edge in this
model class: across five configurations, FiLM-only conditioning stays
action-blind (the hinge saturates — ignoring the plan is the cheaper
equilibrium), while a structural plan-into-trunk pathway does reach
sample-level action-sensitivity (conditioned-over-free error 0.56–0.75;
wrong-plan reconstructions up to 3x worse) but destabilizes to
divergence within thousands of steps at every tested weight and learning
rate, or oscillates without converging. The capability exists; a stable
training equilibrium was not found (candidate stabilizers — EMA,
contrast warmup, x0-parameterization, a sequence-structured trunk —
remain untested). The lesson is the same one §6.1 teaches from the other
side: the binding constraint on thinking-in-frames is the action-causal
quality of the imagination; search strategy and judge quality are
second-order by comparison.

### 6.3 Corrective data: true DAgger ladders, then hits the representation wall (Fig. 7)

Every self-improvement failure above used *outcome-labeled own episodes*.
The classic DAgger guarantee assumes something stronger: an expert's
action at each state the learner visits. Our environment's scripted
expert (41.5% success) makes the real thing testable: roll out the
think-then-act agent; at every visited state, snapshot the environment,
roll the expert forward one horizon to obtain a corrective action chunk,
restore, and continue. The healthy cycle-1 imagination and judge are
*held fixed* — only the BC candidate head retrains each round on the
aggregated (visited state → expert chunk) pairs, isolating the data
variable completely.

It ladders. In both seeds, round 1 alone lifts the best arm past the
campaign's all-time ceiling (8.7% / 9.3% vs 6.1%), round 2 continues
(8.7% / 11.1%), and round 3 flattens (9.8% / 9.7%) — six consecutive
round-evaluations above every number produced by any other method, n =
1000 each. Two boundaries are equally informative. First, think-vs-BC
ordering remains seed- and round-inconsistent (±2pp) even in the
laddering regime: with a strong corrective prior, act-time selection
neither reliably helps nor hurts — consistent with deliberation being a
crutch for a weak policy, shed as the policy improves (Dreamer's
distill-then-drop-the-search endgame, reached from the opposite
direction). Second, the plateau at ~10% against a 41.5% teacher, with BC
action accuracy long known to be encoder-capped (§6, round 8), locates
the next binding constraint in the frozen representation — not in data
quantity, data quality, judge, or search. The campaign's constraint
ordering is thus: (1) action-causal imagination, (2) corrective on-policy
data, (3) representation capacity — with judge and search strategy never
binding at any point we could measure.

### 6.4 A second domain: Doom, a borrowed tokenizer, and a perfect teacher (Fig. 8)

Everything above is one environment. To test whether the stack — and the
constraint ordering — is domain-general, we added ViZDoom
(health-gathering: survive 160 steps on an acid floor by collecting
medikits; a privileged-state scripted oracle survives 100%) behind the
same nine-action environment interface, and deliberately kept the *drone
game's* frozen tokenizer, which has never seen a Doom frame. The planner
trained on these borrowed latents passes the same offline audit gates as
the drone planner (rank-fidelity 0.86–0.92, zero-plan contrast 10.7–14.1x),
and the full expert-DAgger ladder of §6.3 then runs unchanged.

One methodological result first. Our original labeling loop —
snapshot, roll the expert a horizon, restore, *at every step of the live
episode* — silently destroys the episode it is labeling: ViZDoom's
save/load carries a small state perturbation, and 160 accumulated
restores kill the 100% oracle entirely (0/20 survival vs 20/20 clean;
our round-0 "expert" collected 0/200). The fix is structural, not
parametric: expert episodes take chunk labels from the expert's *own
executed future* (no snapshots at all), and agent episodes save
snapshots during the rollout but defer every labeling restore to after
the episode ends. Corrective labeling must never touch the mainline
trajectory it is correcting — a constraint invisible in simulators with
exact state restore, and exactly the kind of defect the drifted run
turns into a free ablation: a *noisy teacher*.

The two ladders (three rounds each, n = 500 per evaluation, two seeds
for the clean run) say three things (Fig. 8). First, the clean teacher
produces the campaign's highest absolute closed-loop performance
immediately: selection reaches 78.6–84.4% survival in every evaluation
across both seeds (vs 41.5%-teacher-capped ~10% in the drone game), and
saturates in round 1 — rounds 2 and 3 add 100k more labeled states and
no selected-policy gains. The representation wall of §6.3 replicates in
a second domain, on a tokenizer that never saw it, just at a far higher
ceiling. Second, the teacher-label channel is decisive: the drift-noised
teacher's best round (45.0%) is barely half the clean teacher's worst
(78.0%) — the poisoning constraint of §6 in its purest form, with the
world model held fixed and only labels varying. Third, selection is a
robustness layer: under aggregation the BC head *degrades* monotonically
in both seeds (66.4→25.0%, 75.8→43.6%) while hard-argmax selection over
the fixed imagination holds the ceiling throughout, and the one
evaluation where the selection gate fails (seed 2, round 1) is precisely
where BC is at its strongest (75.8%). Thinking's margin appears exactly
when — and only when — the proposer is off-ceiling, the same
deliberation-as-crutch asymmetry §6.3 observed from the opposite side.

## 7. Visible thinking traces

A decoder-only, motion-weighted fine-tune of the frozen tokenizer (encoder
untouched) renders imagined futures legibly on game scenes: candidate rows
visibly differ, and a walker advances across imagined frames. Fig. 4 shows
a held-out expert context where the decoded imagination is visibly
action-conditioned and the selection machinery works end-to-end: the
selected candidate's future error (0.0034) nearly matches the true plan's
(0.0028), while a random plan is 5.8x worse and a zeroed plan 16x worse.
Real-robot scenes render layout but lose detail — a decoder-capacity limit,
not a latent one (reconstruction of *real* latents has the same limit), so
quantitative claims on robot data rest on the latent-space audit.

## 8. Related work

**Learning behaviors inside learned world models.** The Dreamer line trains
a policy by backpropagating through (or bootstrapping values within)
imagined latent rollouts of a recurrent state-space model, from DreamerV2's
discrete latents [Hafner et al., 2021 — verify] to DreamerV3's
domain-general recipe — including the stop-gradient, head-only training of
reward and value predictors that we rediscovered as our reward-detach fix
[Hafner et al., 2023 — verify] — and Dreamer 4's scaled, action-conditioned
video tokenizer trained largely from offline video [Hafner et al., 2025 —
verify]. Earlier work planned online against learned latent dynamics with
sampling-based MPC (PlaNet [Hafner et al., 2019 — verify]; PETS [Chua et
al., 2018 — verify]) or hybridized value learning with short-horizon latent
search (TD-MPC2 [Hansen et al., 2024 — verify]). Our setting differs in
that the visual model comes first: like the line of work that retrofits
action interfaces onto frozen video generative models (latent-action and
playable world models such as Genie [Bruce et al., 2024 — verify] and
interactive/neural game engines [Valevski et al., 2024 — verify]), we keep
the tokenizer frozen and ask whether a planner bolted onto its latent space
supports act-time selection. Our two-seed reversal is, in effect, an
empirical argument for Dreamer's core design choice — train a policy in
imagination rather than search over a learned value at act time.

**Evaluating world models as decision-makers.** Video world models are
usually evaluated by generation fidelity (e.g., FVD [Unterthiner et al.,
2018 — verify]) or by reward/return prediction, neither of which measures
whether the model's rankings of *candidate* futures are trustworthy; recent
work has begun probing action controllability and physical consistency
directly [Bruce et al., 2024 — verify; Kang et al., 2024 — verify]. Our
audit contributes external decision-quality proxies — future error of
selected candidates against the realized future, persistence baselines,
oracle rank of the true plan, in-distribution candidate banks — and two
cautionary findings for this literature: training-time selection metrics
computed in the scorer's own score space can be circular, and even
externally-validated offline gates fail to predict the sign of closed-loop
selection value across seeds.

**Offline model-based RL, model exploitation, and action-chunk policies.**
That optimizing against a learned model exploits its errors is a classic
observation: offline model-based methods answer with ensemble-uncertainty
penalties (MOPO [Yu et al., 2020 — verify]; MOReL [Kidambi et al., 2020 —
verify]) or short, truncated rollouts (MBPO [Janner et al., 2019 —
verify]), while model-free offline RL constrains the policy to the behavior
distribution (BCQ [Fujimoto et al., 2019 — verify]; CQL [Kumar et al.,
2020 — verify]; TD3+BC [Fujimoto & Gu, 2021 — verify]). Our BC-anchored
candidate restriction is the search-space analogue of behavior
regularization, and our PMPO instantiation follows the
KL-to-prior-weighted-regression family (MPO [Abdolmaleki et al., 2018 —
verify]). Finally, our planner's action-chunk plans and inverse-dynamics
decoding connect to action-chunking policies (ACT [Zhao et al., 2023 —
verify]; diffusion policies [Chi et al., 2023 — verify]) and to
vision-language-action models that emit chunked actions directly from
context (RT-2 [Brohan et al., 2023 — verify]; OpenVLA [Kim et al., 2024 —
verify]; pi-0 [Black et al., 2024 — verify]). The plan-shortcut symmetry
we document is a hazard specifically for such hybrids: any act-time head
that can read the action content from a plan/context token will route
around the world model unless trained not to.

## 9. Honest claims and limitations

We claim: the offline think-then-act loop closes, repeatably (two seeds),
with transfer of *selection* to a held-out source; timing grounding is
achievable and its data-identifiability is measurable (DROID's shift-1
action cosine of 0.91 makes timing unidentifiable there — a data property);
offline decision-quality metrics do not predict the sign of closed-loop
selection value (two-seed reversal, both CI-clean); and one DAgger iteration
restores a two-seed-consistent behavioral win for thinking (success and
return over both controls, all CIs clear, n = 1000 x 2 seeds). We claim additionally: the self-training poison localizes to the
imagination, not the value head (8/8 exchange cells, two seeds, Fig. 6),
while pre-DAgger seed-fragility is distributed across components (4-cell
exchange); likelihood-only generative proposers recapitulate scene
priors, so value-guided sampling over them fails softly (two seeds, both
judges); soft plan-manifold search does not beat random selection in
either seed; contrast training of the diffusion prior reaches
action-sensitivity only on an unstable knife-edge (five configurations);
and expert-corrective data ladders the stack past its previous ceiling in
both seeds before plateauing at the representation bound (§6.3). We
do not claim: behavioral improvement without on-policy data (refuted by
the reversal); iterated self-improvement from outcome-labeled self-data (a
second self-collection round fails the strict gate under accumulation,
rebalancing, and replacement, and replacement re-inverts selection to
CI-clean worse-than-random); real-drone control; robot-source acting transfer; or long-horizon
(h16) parity — improved from 3.4–28x worse than persistence to 1.7–2.8x,
but open: a horizon-tail curriculum did not close it (expert 3.12 / SOAR
6.98 at h16; held-out Bridge improved 11.8 → 6.7, with h8 at 0.86 — below
persistence on the held-out source for the first time), and that run is
confounded by a regressed lineage, so the gate is not passed and a clean
retest from the headline checkpoint remains open.

## 10. Future work

Policy-in-imagination with a control-purposed encoder (the identified root
constraint); source-balanced ranking to remove the scorer transient; online
iteration to close the remaining distribution gap. A first lightweight
instantiation of the policy route (offline PMPO in the planner's imagination
against its plan-free value, with a KL to a built-in BC prior whose gradients
shape the context encoder; two seeds at n = 1000) learned survival (72–76
steps, few collisions) but not goal-reaching (~0% success), with
return-vs-BC flipping sign across seeds and thinking adding nothing over the
policy alone — consistent with our central finding, and indicating that
behavioral closure requires online interaction or substantially longer
imagination-RL rather than offline-only training. For making the DAgger
repair iterate, the cycle-2 arc identifies the levers: success-preserving
replay (the improved agent's rare successful episodes are the diluted
signal) or expert-corrected relabeling of visited states, in place of raw
outcome-labeled self-imitation — and §6.1 says these interventions should
be aimed at the *dynamics* training signal, not the value head. For soft
deliberation, §6.2 makes the prerequisite explicit: an action-causal
generative proposer (contrast-trained, as the GRU proposer was — not
likelihood-only), after which guidance has something to steer; a
plan-token variant (gradient ascent on the judge through the
action-conditioned proposer, re-projected onto the unit-norm plan sphere
each step) was run and does not beat random selection (§6.2); the §6
seed-reversal stacks have been decomposed (§6.1). What remains open is
stabilizing contrast-diffusion training (EMA, contrast warmup,
x0-parameterization, a sequence-structured trunk) — the engineering
problem standing between this paper's negative and a working
soft-thinking substrate.

## 11. Reproducibility

All experiments run in a single container; every training run, audit, and
eval is a script with a logged manifest.

**Scripts** (all under `scripts/`): `train_latent_imagination_planner.py`
(planner + all four fixes as flags),
`eval_latent_imagination_decision_quality.py` (the external audit),
`eval_act_by_imagination.py` (offline acting gates),
`eval_gym_drone_game_act_by_imagination.py` (closed-loop, matched seeds,
paired bootstrap), `collect_gym_drone_game_dreamer4_dataset.py`
(`--pad-terminal` absorbing states), `collect_gym_drone_game_branch_dataset.py`
(env snapshot/restore counterfactuals), `relabel_rewards_return_to_go.py`,
`train_drone_bc_chunk_head.py` (behavior prior),
`train_drone_imagination_policy.py` (offline PMPO),
`collect_dagger_episodes.py` (the DAgger loop-closer),
`decode_imagination_traces.py` + `finetune_tokenizer_decoder.py`
(visible thinking), `make_paper_figures.py`.

**Headline checkpoints** (under `output/`): offline —
`latent_imagination_planner_all_data_v3_rankfix_armE_seed2/planner_ckpts/final.pt`;
robot offline acting — `..._all_data_v5_crossinv/planner_ckpts/final.pt`;
drone closed-loop (post-DAgger, both seeds) —
`..._drone_game_v8_dagger_c1{,_seed2}/planner_ckpts/final.pt` with
`drone_bc_chunk_head_dagger_c1{,_seed2}.pt`.

**Key result records** (under `output/`): decision-quality audits
(`decision_quality_audit_*`); closed-loop evals
`closed_loop_drone_game_v9_power` (seed-1 win), `v10_power_seed2`
(reversal), `v11_pmpo_s2026090{1,2}` (PMPO), `v12_dagger_c1{,_seed2}`
(post-DAgger consistency). **Source docs:**
`DECISION_QUALITY_AUDIT_RESULTS.md`, `ACT_BY_IMAGINATION_HARNESS.md`,
`WORKLOG.md`.

---
*Figures: output/paper_figures/fig0_architecture.png,
fig1_training_dynamics.png, fig2_closed_loop.png (four-panel arc:
pre-DAgger win/reversal, post-DAgger two-seed consistency),
fig3_two_seed_scorer.png, fig4_trace_grid.png, fig5_dagger_cycles.png
(the iteration arc: repair, not ladder), fig6_decomposition.png (the
exchange test + guided-diffusion floors), fig7_expert_dagger.png
(corrective data ladders); full trace grids under
output/imagination_traces_armE_latest_v2dec/.*
