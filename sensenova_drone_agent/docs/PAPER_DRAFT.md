# From Scene Priors to Decision-Quality Imagination: Retrofitting Action Grounding into Video World-Model Latent Spaces

*Draft v1 — 2026-07-10. Numbers are final from the July 2026 campaign; see
WORKLOG.md for artifact paths. Bracketed notes mark writing TODOs.*

## Abstract

Video world models promise agents that can *think in frames*: imagine
candidate futures under different actions, evaluate them, and act on the
best one. We study this think-then-act loop directly, on top of a frozen
video tokenizer, using a latent imagination planner that proposes
action-conditioned futures, scores them, and inverts selected futures into
actions. Our first contribution is negative and methodological: the
standard training-time selection metric is circular, and under an external
audit — future-error against the real future, persistence baselines, and
in-distribution candidate banks — the planner's scorer was almost perfectly
*anti-correlated* with reality on robot data (r = −0.97) while the training
metric read healthy. We introduce a decision-quality audit and use it to
drive four verified fixes — candidate-ranking supervision, unit-norm plan
tokens, reward-path stop-gradients, and per-step plan conditioning — which
close the offline loop: candidate selection beats random on every trained
source and transfers to a never-trained source (98–100% of contexts), while
a reproducible phase transition grounds exact action timing (wrong-timing
error ratios rising from 1.0 to 11–22x). We further show a symmetric
shortcut failure: *any* act-time head given the plan token learns to route
around imagination (the inverse-dynamics head decodes actions from the plan;
the scorer becomes Q(state, plan)); plan-dropout training is the general
cure, demonstrated twice. Finally, we evaluate the loop in a closed-loop
drone navigation game across nine controlled rounds, contributing a
structural dataset finding (window enumeration never places terminal
transitions inside scored futures), a diagnosis of offline-MPC model
exploitation, and a two-seed behavioral test whose outcome sharpens that
diagnosis: selecting among behavior-prior candidates by imagined value beats
acting without thinking in one training seed (success +1.4pp CI [+0.4,+2.6];
return +1.74 CI [+1.20,+2.25]; n = 1000 matched episodes) and *reverses,
CI-clean, in a second seed whose offline metrics are equal or better* —
offline decision-quality gates do not predict even the sign of act-time
selection value. We release the audit, harnesses, and counterfactual
branching tools. [~200 words; trim to venue limit]

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
   concentrates on timing negatives — wrong-timing error ratios jump from
   1.0 to 5–22x within ~20k steps, in both seeds, with onset tracking the
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
   return with CIs clear of zero at n = 1000.

Throughout we maintain a strict claim discipline (§8): no claims about real
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
8, horizon 8, at 128×128 resolution. [architecture figure TODO]

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
path* drags imagined futures toward score targets — proposer fidelity
collapsed 5–10x in both arms. **Stop-gradients on all scorer inputs in the
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

**Final offline results** (seed 2, 210k): all three game sources pass 6/6
gates (timing 11–27x, selection beats random 87–93%); SOAR 5/6 (timing
grounded at 5.0, selection 87%); held-out Bridge 5/6 with selection beating
random in 100% of contexts. The single failing robot gate everywhere is the
true-vs-shuffle *score margin*. [Table 1: per-source gate table]

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
error) relative to the oracle ceiling — the 128-d plan bottleneck —
and Bridge transfer remains open.

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

Nine rounds, each removing one verified defect [Table 2: round table]:
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
positive (Fig. 2): 2.8% success and −0.42 return for think-then-act versus
1.4% / −2.16 for the same BC acting without thinking (success +1.4pp,
CI [+0.4, +2.6]; return +1.74, CI [+1.20, +2.25]), with directed
goal-reaching versus random selection (+1.78 return; 25.7 vs 37.2 steps).
A full-stack repeat with an independent training seed **reversed every
comparison with CIs clear of zero** (think 0.9% vs BC 2.2% success; return
−2.26 vs BC) — despite the repeat seed's *offline* metrics being equal or
better (scorer fidelity +0.78 vs +0.57; identical BC accuracy). We
therefore do not claim a behavioral improvement. What the two-seed test
establishes instead is stronger than the positive would have been: offline
decision-quality metrics, even in-domain and near ceiling, do not predict
the **sign** of act-time selection value; the value head's act-time
preferences are seed-fragile in ways invisible to every offline gate. This
is the sharpest form of our central methodological finding and the direct
motivation for policy-in-imagination over act-time value search (§9).

## 7. Visible thinking traces

A decoder-only, motion-weighted fine-tune of the frozen tokenizer (encoder
untouched) renders imagined futures legibly on game scenes: candidate rows
visibly differ, and a walker advances across imagined frames.
[Fig. 4: trace grid from output/imagination_traces_armE_latest_v2dec/]
Real-robot scenes render layout but lose detail — a decoder-capacity limit,
not a latent one (reconstruction of *real* latents has the same limit), so
quantitative claims on robot data rest on the latent-space audit.

## 8. Honest claims and limitations

We claim: the offline think-then-act loop closes, repeatably (two seeds),
with transfer of *selection* to a held-out source; timing grounding is
achievable and its data-identifiability is measurable (DROID's shift-1
action cosine of 0.91 makes timing unidentifiable there — a data property);
and offline decision-quality metrics do not predict the sign of closed-loop
selection value (two-seed reversal, both CI-clean). We do not claim: any
closed-loop behavioral improvement from thinking (single-seed only, not
robust); real-drone control; robot-source acting transfer; or long-horizon
(h16) parity — improved from 3–28x worse than persistence to 1.7–2.8x, but
open.

## 9. Future work

Policy-in-imagination with a control-purposed encoder (the identified root
constraint); source-balanced ranking to remove the scorer transient; online
iteration to close the remaining distribution gap. A first lightweight
instantiation of the policy route (offline PMPO against the frozen planner's
imagination, KL to a BC prior, two seeds at n = 1000) learned collision
avoidance but not goal-reaching, with seed-inconsistent returns — consistent
with our central finding, and indicating that behavioral closure requires
online interaction or substantially longer imagination-RL rather than
offline-only training.

## 10. Reproducibility

All experiments run in a single container; every training run, audit, and
eval is a script with a logged manifest. [artifact list = §10 of
PAPER_SKELETON.md; consolidate on release]

---
*Figures: output/paper_figures/fig1_training_dynamics.png,
fig2_closed_loop.png, fig3_two_seed_scorer.png; trace grids under
output/imagination_traces_armE_latest_v2dec/.*
