# Decision-Quality Audit Results (130k Latent Imagination Planner)

Date: 2026-07-06

Audit script: `sensenova_drone_agent/scripts/eval_latent_imagination_decision_quality.py`
Launcher: `sensenova_drone_agent/scripts/experiments/launch_decision_quality_audit.sh`
Checkpoint: `latent_imagination_planner_all_data_v1/planner_ckpts/latest.pt` (step 130000)
Outputs:

```text
sensenova_drone_agent/output/decision_quality_audit_130k_v1/          8 sources x 256 contexts
sensenova_drone_agent/output/decision_quality_audit_ckptsweep_00{5,9,13}0000/  50k/90k/130k on soar+expert
```

This is the first measurement of the full think-then-act loop with an EXTERNAL
proxy (future-MSE to the real future) instead of the scorer's own outputs. The
training-time metric `candidate_selected_minus_random` was circular (argmax of
scores minus a random score, in score space) and never measured decision
quality.

## Verdict In One Paragraph

The imagination half of the loop works at short horizon on trained sources:
true-action plans produce futures that beat persistence at h4/h8 and beat
zero/shuffle action variants by large margins (zero ratio 4-12x, shuffle up to
1.7x). The evaluation half of the loop is broken: the trajectory scorer ranks
the true plan in the bottom half of candidates almost everywhere, prefers
zero-action plans on all robot sources (score margin -3 to -5.5), and on
SOAR/bridge is almost perfectly ANTI-correlated with fidelity (r = -0.97).
Score-based candidate selection is therefore no better - and on robot sources
strictly worse - than picking a random imagined future. No source passes all
gates. Checkpoint sweep 50k->90k->130k shows the failure is stable, so more
training steps under the current objective cannot fix it.

## Gate Results (130k, 256 contexts/source)

```text
source                          zero   shuffle tshift  fid_corr  sel-rand@k64  verdict
                                (mse ratio, >1 good)   (bank)    (mse, <0 good)
dreamer4_hf_expert              4.33   1.23    1.03    +0.02     +0.0010       scorer flat
dreamer4_hf_mixed_small         5.63   1.14    1.02    -0.08     +0.0026       scorer harmful
dreamer4_hf_mixed_large         6.28   1.22    1.00    -0.17     +0.0029       scorer harmful
soar_native_v2                  10.19  1.71    1.01    -0.97     +0.0328       scorer inverted
hf_robot_droid                  11.65  1.00    1.00    +0.15     +0.0004       actions ignored beyond zero
hf_robot_fractal                1.91   1.01    1.01    -0.40     +0.0147       scorer inverted
hf_robot_bridge (HELD OUT)      2.59   1.74    1.04    -0.96     +0.0114       scorer inverted
robonet_sample_64               1.06   1.05    1.01    -0.27     -0.0040       spurious pass (tiny source)
```

## Finding 1: The scorer is the broken component

- `oracle_rank_pct` of the true plan among candidates: 0.18 (SOAR) to 0.65
  (expert). `oracle_top1` is 0.00 on every robot source.
- On every robot source the zero-action plan OUTSCORES the true plan
  (score margin -3.0 to -5.5), even though its imagined future is 2-12x
  farther from reality.
- Fidelity correlation on in-distribution (bank) candidates: -0.97 on SOAR,
  -0.96 on held-out bridge, -0.40 fractal. The scorer is not merely untrained
  on candidates; it has learned an artifact that inverts fidelity.
- Root cause in `train_latent_imagination_planner.py`: the scorer is trained
  ONLY as return regression on the true plan's imagined future
  (`score_future(ctx_h, pred_future(true_plan), true_plan)` vs discounted
  return). It never sees a wrong or sampled plan during training, so ranking
  candidates at inference is pure extrapolation.
- Return targets: DROID / fractal / bridge exports have literally zero reward
  everywhere (return std 0.0000), so on those sources the regression target is
  degenerate. SOAR has usable return variance (std 3.9, 45% nonzero) but the
  scorer still learned nothing there (corr -0.05); only the dreamer4-HF game
  sources produced a working return head (corr 0.57-0.78). The 0.61
  score_return_corr in the training logs was carried by game data.

## Finding 2: The proposer thinks, but coarsely

- Action-causal vs zero/shuffle in future-MSE space on trained sources
  (see table). This is real thinking signal: contemplated actions change the
  imagined future in the right direction.
- Exact timing is NOT grounded: time_shift / time_shift2 / time_perm ratios
  are 1.00-1.09 everywhere, at every checkpoint. The absolute contrast hinge
  margin (0.02) is ~3x the typical normal future-MSE (~0.007), so the hinge is
  saturated and the gradient pressure is shared with the much easier
  zero/shuffle negatives.
- Horizon: better than persistence at h4/h8 on trained robot sources
  (0.52-0.94), collapses at h16 (3.4-5.8x worse than persistence). Imagination
  is trustworthy for ~8 steps only.
- Held-out bridge: worse than persistence at ALL horizons (3.2/5.6/19x) while
  still action-sensitive (zero 2.6, shuffle 1.7). The proposer's absolute
  fidelity does not transfer to unseen sources, but relative action
  sensitivity partially does.
- DROID: shuffle ratio exactly 1.00 - swapping another context's actions
  changes nothing. Matches the earlier DROID identifiability audit (weak
  incremental action signal). Only the zero/no-op distinction is learned
  there.

## Finding 3: Plan space is not samplable

- Encoded plan norms: 69-137 vs 11.3 expected for N(0,1) samples of dim 128.
  Training-eval's randn candidates were ~10x out of distribution; their
  proposals behave like the zero-plan token.
- `matched` Gaussian candidates fixed the scale but not the manifold: their
  fidelity correlation is ~0 everywhere. Only `bank` candidates (real encoded
  plans from other contexts) expose the scorer's anti-correlation clearly.

## Checkpoint Sweep (50k / 90k / 130k)

Decision metrics are flat or noisy across checkpoints (SOAR fid_corr -0.90,
-0.91, -0.96; oracle_rank_pct 0.17, 0.16, 0.19). The scorer failure is a
stable property of the objective, not an undertrained head. Do not resume the
current objective expecting decision quality to emerge.

## What This Means For The Think-Then-Act Loop

```text
thinking (imagine action-conditioned futures):   partial pass (h<=8, coarse timing)
evaluating thoughts (score candidate futures):   FAIL (anti-correlated on robot data)
acting on thoughts (select-then-invert):         blocked by the scorer
```

The next experiment must train the scorer ON candidates, not only on the true
plan, and must make plan space samplable by construction.

## Experiment 2 Design (v2 "rankfix")

Trainer changes (`train_latent_imagination_planner.py`):

1. Candidate-ranking loss: per batch item, build a candidate pool
   (true plan, control-variant plans, bank plans from other batch items,
   matched-Gaussian samples), roll them out, and train the scorer with a
   pairwise hinge so that candidates whose imagined future is closer to the
   real future must score higher. This supervises ranking without depending
   on reward labels, so it works on DROID/bridge too. Return regression stays
   as a secondary loss where reward variance exists.
2. Plan normalization: L2-normalize plan tokens to radius sqrt(plan_dim) so
   sampled candidates are in-distribution by construction and the scorer
   cannot key on plan norm.
3. Relative contrast margin: replace the absolute hinge margin (0.02) with a
   ratio margin (neg_mse >= r * normal_mse, r ~= 1.2) so the timing negatives
   (time_shift/perm/reverse) keep receiving gradient after the easy negatives
   saturate.

Two arms to isolate the plan-norm change:

```text
arm A (GPU 1): rank loss + relative margin, resume from 130k
arm B (GPU 0): rank loss + relative margin + unit-norm plans, resume from 130k
```

Re-audit both arms with eval_latent_imagination_decision_quality.py on
soar_native_v2 + dreamer4_hf_expert + held-out bridge. Success criteria:
bank fidelity_corr CI > 0, sel-minus-rand mse CI < 0 and improving with K,
true plan oracle_rank_pct > 0.8, time-shift mse ratio >= 1.05.

## Round 1 Results (arms A/B at 150k, 2026-07-07)

Audits: `output/decision_quality_audit_armA_150k/`, `output/decision_quality_audit_armB_150k/`.

The ranking loss fixes the scorer, and unit-norm plans are the right geometry:

```text
bank fidelity_corr           130k      armA 150k   armB 150k (unit-norm)
soar_native_v2               -0.97     -0.35       +0.64
dreamer4_hf_expert           +0.02     +0.41       +0.23
bridge (HELD OUT)            -0.96     -0.64       +0.38
```

Arm B selection beats random 83% of the time on held-out bridge
(sel-minus-rand -0.049, 5/6 gates) and the true plan outscores the zero plan
on every source (SOAR margin flipped -4.2 -> +4.2). Arm A passed 6/6 gates on
expert but stayed fidelity-inverted on SOAR/bridge: without unit-norm the
norm artifact persists.

BUT both arms' proposers collapsed: true-plan future MSE went from
0.7-1.5x persistence to 8-18x persistence, and train future_loss rose 5-10x.
Root cause: `pred_score = score_future(ctx_h, pred_future, plan)` let reward
regression backprop into the proposer. Harmless in v1 (scorer insensitive to
future content); once the rank loss made the scorer fidelity-sensitive, the
reward gradient dragged imagined futures away from reality. Fixed by
detaching all scorer inputs in the reward-regression path (scorer-head-only
training, Dreamer-style stop-gradient).

## Round 2 (arms C/D, from 130k, reward-detach + unit-norm + rank loss)

```text
arm C (GPU 1): relative contrast margin 1.2
arm D (GPU 0): absolute contrast margin 0.02 (isolates the margin change;
               timing ratios stayed ~1.0 in round 1 either way)
```

Watch future_mse in eval rows: it must stay near v1 levels (~0.01) this time.
Timing grounding (time_shift ratio ~1.0) is still unsolved in all arms and
likely needs data-side work (contrastive thinking pairs), not more margin.

## Round 2 Results (arms C/D at 150k, 2026-07-07)

Audits: `output/decision_quality_audit_armC_150k/`, `output/decision_quality_audit_armD_150k/`.
The reward-detach fix held: train future_mse stayed at 0.005-0.008 all round.

- Arm C (relative margin): high fidelity (0.44-0.56x persistence, even on
  held-out bridge) but achieved it by collapsing plan sensitivity: zero ratio
  1.2, shuffle 1.00, candidate futures near-identical (sel-rand ~ 0.0000).
  Accurate but plan-independent futures - no thinking diversity. Structural:
  the relative margin releases contrast pressure once negatives clear 1.2x.
  Arm killed at 150k; checkpoints retained.
- Arm D (absolute margin): best full loop so far. 6/6 gates on expert
  (fid_corr +0.42, selection beats random 80%, monotone K-improvement),
  5/6 SOAR (fid +0.51), 4/6 held-out bridge (fid +0.67, beats random 93%).
  Proposer action-sensitivity intact (zero 2.6-3.4) but fidelity sits at
  1.8-3.6x persistence (baseline 0.7-1.5) - a fidelity/sensitivity trade-off
  paid for keeping contrasts wide. Timing still ~1.0. Continues to 210k.

## Timing Identifiability Check (raw actions, shift-1 cosine per window)

```text
dreamer4_hf_expert   0.33   timing clearly identifiable in data
bridge (held out)    0.49   identifiable
soar_native_v2       0.78   partially identifiable
droid                0.91   near-unidentifiable (explains droid tshift=1.0)
```

Expert actions decorrelate fast yet tshift stays ~1.0 there in every arm and
at every checkpoint -> the timing failure is ARCHITECTURAL, not (only) data:
the proposer consumes one static plan token and no pathway ties rollout step
t to the action at step t.

## Round 3: arm E (per-step plan conditioning)

`plan_step_head: Linear(plan_dim, horizon*hidden)` decodes the plan token
into H per-step embeddings added to the rollout cell input at each step.
Plan stays a single samplable token; the architecture gains a timing pathway.
Config: arm D stack (rank loss, unit-norm, absolute margin, reward-detach)
+ `--plan-step-conditioning`, resumed non-strict from 130k (new params fresh,
optimizer state reset). GPU 1, out dir
`latent_imagination_planner_all_data_v3_rankfix_armE`.

Watch: time_shift/time_shift2 over normal moving off 1.0 on expert-heavy
evals is the success signal; also verify future_mse and zero ratio stay at
arm-D levels. Compare arm E vs arm D (control) at 150k-160k with the audit.

## Round 3 Results (armE 150k vs armD 150k/174k, 2026-07-07)

Audits: `output/decision_quality_audit_armE_150k/`, `_armD_174k/`.

### Timing phase transition

Arm D (control) ground timing WITHOUT the architecture change, via a slow
phase transition starting ~148-151k: eval tshift 1.0 -> 5.3 over ~18k steps.
Mechanism: the ABSOLUTE hinge margin (0.02) implicitly anneals - as normal
future-MSE fell to ~0.005 the margin demanded negatives ~5x worse, and once
zero/shuffle saturated that bar, gradient concentrated on timing negatives.
(The relative margin of arm C can never do this - it releases at fixed 1.2x.)

Arm E (per-step plan conditioning) hit the same transition ~7k steps earlier
(tshift 1.74 at 144k vs arm D 0.98 at 144k) and reached better overall
quality at matched step count.

### Arm E at 150k is the strongest checkpoint of the project

```text
                       expert          SOAR            bridge (HELD OUT)
gates                  6/6 ALL PASS    5/6             5/6
bank fid_corr          +0.50           +0.59           +0.89
sel beats random @k64  91%             88%             98%
zero mse ratio         11.95           22.96           18.40
shuffle mse ratio      7.97            1.52            2.48
tshift mse ratio       2.92            1.03            1.12
persist h4/h8          0.43/0.40       0.49/0.63       1.16/1.60
```

vs arm D at matched 150k: arm E wins on essentially every axis (arm D 150k:
expert tshift 1.01, persist 1.84/1.89, zero 2.55). Per-step conditioning did
not just accelerate timing - it RESOLVED the fidelity/action-sensitivity
trade-off (fidelity better than the 130k baseline AND causality far
stronger). Arm D at 174k catches up on expert timing (tshift 5.64) but still
trails arm E-150k on bridge/SOAR fidelity and selection.

### Think-then-act loop status after round 3

```text
thinking (imagine action-conditioned futures):  PASS on trained sources
  (beats persistence h4/h8, zero/shuffle/timing causal on identifiable data)
evaluating thoughts (score candidates):         PASS
  (fid_corr +0.5 to +0.9 incl. held-out source, selection beats random
   88-98% with monotone K-improvement)
acting on thoughts:                             ready for downstream use
```

Remaining gaps:
- true_beats_shuffle SCORE-margin gate fails on SOAR/bridge (the futures
  differ, the scorer just does not confidently prefer true over shuffled
  plans there).
- timing ratios stay ~1.0-1.35 on SOAR/bridge - consistent with partial data
  identifiability (shift-1 action cosine 0.78/0.49) vs expert (0.33).
- h16 open-loop still worse than persistence everywhere (4.6-5.5x).

Both arms continue to 210k; final audit + repeatability seed to follow.

## Arm D Final Audit (210k) — continued compounding

Audit: `output/decision_quality_audit_armD_210k/`.

```text
                 expert            SOAR              bridge (held out)
tshift ratio     22.10             3.59              1.88
shuffle ratio    24.29             1.34              1.61
zero ratio       17.45             37.29             28.99
persist h4/h8    0.28/0.40         0.18/0.21         0.21/0.36
persist h16      2.81              1.69              2.27
gates            6/6               4/6               5/6
```

SOAR timing moved off 1.0 (3.59) once training continued past the phase
transition — partial data identifiability (shift-1 cosine 0.78) is now being
exploited. Held-out bridge fidelity is better than persistence at h4/h8.
h16 nearly closed (1.7-2.8x, from 3.4-28x). Remaining fail: true-vs-shuffle
SCORE margin on SOAR/bridge.

## Arm E Final Audit (210k, all 8 sources) And The Robot-Scorer Instability

Audits: `output/decision_quality_audit_armE_210k_full/`, `_armE_0170000_robot/`, `_armE_0190000_robot/`.

Game sources at 210k: complete sweep. expert / mixed_small / mixed_large all
6/6 gates, tshift 11-15x, zero 10-13x, beats-random 92-93%, persistence
ratios 0.30-0.41. Strongest planner results of the project.

Robot sources: the SCORER (not the proposer) destabilized late in training:

```text
armE on SOAR / bridge   fid_corr          beats-random     proposer persist h4
150k                    +0.59 / +0.89     88% / 98%        0.49 / 1.16
170k                    +0.54 / +0.83     77% / 98%        0.36 / 0.95
190k                    -0.51 / -0.56      9% / 10%        0.23 / 0.22
210k                    -0.01 / +0.30     50% / 53%        0.36 / 0.62
```

The proposer kept improving monotonically; the scorer INVERTED on robot
sources between 170k and 190k and only partially rebounded. Mechanism
hypothesis: after the timing phase transition, game-source fidelity spreads
explode, rank-hinge pairs concentrate there, and robot-source score functions
lose their gradient anchor (rank_mse_gap pairs stop firing where spreads are
small). Fix for a future run: source-balanced ranking - normalize hinge
contributions per source (or scale rank_mse_gap by per-source fidelity
spread) so no source's pairs dominate.

### Repeatability (seed 20260777, same config from 130k)

Audit: `output/decision_quality_audit_armE_seed2_150k/`. At 150k, seed 2
matches or exceeds seed 1 on every gate metric — identical gate pattern
(6/6 expert, 5/6 SOAR, 5/6 bridge), fid_corr +0.46/+0.70/+0.90, selection
beats random 82-94%, timing phase transition reproduced on schedule. The
arm E result is NOT a lucky seed.

Seed-2 at 170k (`output/decision_quality_audit_armE_seed2_0170000_robot/`):
the robot-scorer inversion REPRODUCES but arrives earlier — seed 2 is already
inverted on SOAR (fid -0.45) and degraded on bridge (-0.09) at 170k, while
its expert results are pristine (6/6, tshift 13x) and its robot proposer is
the best measured (persist h4/h8 0.18-0.25). Seed 2's phase transition also
started earlier (tshift 3.1 at 148k vs seed 1 ~1.8); the scorer window closes
tracking the transition, not an absolute step count.

### Seed-2 complete trajectory: the oscillation RECOVERS

Full two-seed table (fid_corr / beats-random / tshift / persist-h8 / gates):

```text
SOAR      seed1: +0.59 88% (150k) -> -0.51 9% (190k) -> -0.01 50% (210k)
          seed2: +0.69 82% (150k) -> -0.44 12% (170k) -> +0.33 87% tsh 4.99 (210k)
BRIDGE    seed1: +0.89 98% (150k) -> -0.55 10% (190k) -> +0.30 53% (210k)
          seed2: +0.90 94% (150k) -> -0.09 32% (170k) -> +0.66 100% tsh 2.36 (210k)
EXPERT    both seeds 6/6 at every audited checkpoint; seed2 210k tshift 27x
```

The robot-scorer dip is a TRANSIENT tied to the timing phase transition, and
it recovers: seed 2's final checkpoint is the best overall artifact of the
project — expert 6/6 (tshift 27), SOAR 5/6 (beats-random 87%, timing 4.99),
held-out bridge 5/6 with selection beating random 100% of contexts. Seed 1
at 210k was mid-rebound; its trajectory suggests it would also recover with
more steps. The only failing gate anywhere at seed2-210k is the
true-vs-shuffle SCORE margin on SOAR/bridge.

### Checkpoint recommendation

```text
headline artifact:    armE_seed2 final (210k) - best on all three fronts
robot window backup:  armE step_0170000.pt (seed 1) / step_0150000.pt (seed 2)
control/ablation:     armD final (210k)
```

Portable rule remains metric-based checkpoint selection per seed (the dip
window varies with the transition timing), and the source-balanced rank loss
is the future-run fix to remove the dip entirely.

## Visible Thinking Traces (decoder fine-tunes v1/v2)

- `scripts/finetune_tokenizer_decoder.py`: decoder-only fine-tune, encoder
  frozen (latent space untouched), ckpt format loader-compatible.
  v1 (uniform recon): background sharpens, moving task detail still lost.
  v2 (`--motion-weight 20`, resumes v1): renders moving structure; best
  recon MSE of all three decoders on expert AND SOAR.
- `scripts/decode_imagination_traces.py`: per-context grids
  (real / recon / true-plan / selected / random / worst / zero-plan)
  with score+fidelity sidecar (traces.json).
- Result (`output/imagination_traces_armE_latest_v2dec/`): on dm_control-style
  expert scenes the imagined futures are visibly legible (walker rendered at
  advancing poses, candidate rows visibly differ). On real-robot video (SOAR/
  bridge) scene layout/colors render but fine detail washes out - decoder
  capacity (d_model 128, depth 4), not a latent problem; recon of real
  latents has the same limit. Paper figures should use expert-style scenes,
  with latent-space metrics carrying the quantitative claim everywhere.
