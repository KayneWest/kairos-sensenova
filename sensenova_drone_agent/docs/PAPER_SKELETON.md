# Paper Skeleton — From Scene Priors to Decision-Quality Imagination

Working title options (per the original handoff, still accurate):

1. From Scene Priors to Decision-Quality Imagination: Retrofitting Action
   Grounding into Video World-Model Latent Spaces
2. Auditing and Training Action-Grounded Imagination in Video World-Model
   Latent Spaces

Thesis sentence: a latent imagination planner on top of a frozen video
tokenizer can learn to THINK — propose action-conditioned futures, evaluate
them with a fidelity-grounded scorer, and select better-than-random plans,
transferring to a held-out source — provided four specific shortcut/objective
failures are removed; and the remaining gap between offline decision quality
and closed-loop behavioral control is measurable, diagnosable, and points at
policy-in-imagination as the necessary next ingredient.

## 1. Introduction

- Motivation: video world models as "thinking in frames"; the think-then-act
  loop (imagine candidate futures -> evaluate -> select -> act).
- Contribution list:
  C1. A decision-quality audit with EXTERNAL proxies (future-MSE vs the real
      future, persistence baselines, bank candidates) that exposed a
      training-metric circularity: the standard candidate_selected_minus_random
      score-space metric read +0.19 while true selection was WORSE than random.
  C2. Four verified fixes that close the offline think-then-act loop:
      candidate-ranking loss, unit-norm plan tokens, reward-path detachment,
      per-step plan conditioning (+ the plan-shortcut symmetry: any act-time
      head given the plan token routes around imagination; plan-dropout is
      the general cure — shown twice, for the inverse head and the scorer).
  C3. A reproducible timing phase transition driven by absolute-margin
      annealing, grounding exact action timing (time-shift ratios 1.0 -> 22x).
  C4. Closed-loop behavioral evaluation in a drone navigation game with
      matched-seed paired controls, including a structural dataset finding
      (terminal transitions can never appear in scored windows) and an honest
      negative: offline decision-quality gates do not predict act-time search
      behavior (offline-MPC model exploitation).

## 2. Background / Setup

- Frozen tokenizer (Dreamer4-style, d_model 128) over mixed sources:
  dm_control-like game data (dreamer4-HF expert/mixed), SOAR, DROID, fractal,
  bridge (HELD OUT, manifest weight 0), robonet.
- Planner: plan encoder (ctx, action-chunk) -> 128-d token; GRU future
  proposer; trajectory scorer; inverse dynamics; contrast losses.
  (fig: architecture with the four fixes annotated)
- Gates battery: zero/shuffle/time-shift/perm/reverse mse ratios,
  persistence baselines, oracle rank, fidelity corr, selection-vs-random
  with K-sweeps, bootstrap CIs. All numbers: output/decision_quality_audit_*/.

## 3. The Audit: measuring the think-then-act loop (results = failure map)

- Baseline 130k planner: proposer partially works (zero 4-12x, persist
  h4/h8 < 1 on trained sources) but scorer is fidelity-INVERTED on robot
  sources (bank fid_corr -0.97 SOAR, -0.96 held-out bridge), prefers
  zero-action plans (score margin -3..-5.5), selection worse than random
  (0% beats-random on SOAR/bridge). Checkpoint sweep 50k/90k/130k: stable ->
  not undertraining. Root causes: scorer never sees candidates; return
  targets degenerate on robot exports; plan norms 69-137 vs 11.3 sampled.
- Table 1: per-source gate table at 130k (from DECISION_QUALITY_AUDIT_RESULTS.md).
- The circular-metric finding as a methodological warning.

## 4. Fixing the loop (each fix = ablation arm, all runs documented)

- Arm A/B (rank loss +/- unit-norm): scorer fixed (fid -0.97 -> +0.64), but
  latent reward-path bug exposed -> proposer collapse 5-10x (the reward
  gradient drags imagination once the scorer is fidelity-sensitive).
- Arm C/D (reward-detach; relative vs absolute margin): relative margin
  collapses plan-sensitivity (accurate but plan-independent futures);
  absolute margin implicitly ANNEALS (margin/normal-mse grows as fidelity
  improves) -> timing phase transition at ~150k (tshift 1.0 -> 5.3 -> 22x).
- Arm E (per-step plan conditioning): transition ~7k steps earlier, resolves
  fidelity/sensitivity trade-off; final: 6/6 gates on all game sources,
  selection beats random 92-93%, tshift 11-15x.
- Repeatability: seed 2 reproduces gates, magnitudes, transition; robot-
  scorer post-transition oscillation reproduces (timing tracks transition,
  not steps) and RECOVERS - seed2 final: held-out bridge 5/6 with
  100% beats-random. Portable rule: metric-based checkpoint selection.
- Fig: two-seed trajectory table; fig: fid_corr and tshift vs steps with
  transition marked.

## 5. Acting on thoughts (offline)

- act-by-imagination harness: selected-candidate actions vs random/zero/
  mean-action/repeat-last + oracle ceilings.
- Plan-decoder finding (inverse head reads actions out of the plan token;
  verified by future-shuffle invariance) -> plan-dropout + imagined-future
  inverse -> cross-context imagined-inverse (context cannot explain the
  target; only the future can).
- Result: 5/5 acting gates on expert (sel-minus-rand CI < 0, beats
  mean-action prior, monotone in K), 4/5 SOAR; margins thin (plan-dim
  bottleneck); no bridge transfer. Table from ACT_BY_IMAGINATION_HARNESS.md.

## 6. Closed-loop behavioral evaluation (drone game)

- Setup: 900 mixed-policy episodes, planner trained in-domain (offline
  metrics near-ceiling: ret_corr +0.93, fid +0.94, zero 100x+).
- Six rounds, each removing a verified defect:
  R1 inverse-head discrete-argmax collapse (mean-action trap, discrete form)
  R2 action-native MPC -> scorer myopia (8-step shaped return)
  R3 RTG value targets -> identical behavior -> exposes R4
  R4 STRUCTURAL: WMDataset terminals never inside scored windows; absorbing
     padding fix
  R5 scorer plan-shortcut (the imagination correctly predicts the crash as
     frozen frames; the scorer ignores it) -> score plan-dropout -> first
     behavioral movement (survival 2.4x, paired return CI > 0)
  R6 true counterfactual branch data (env snapshot/restore; the handoff's
     "contrastive thinking data") -> offline gains, closed-loop regression.
- Diagnosis: offline-MPC model exploitation under distribution shift;
  offline gates cannot certify act-time search. Why Dreamer trains a policy
  in imagination.
- BC-anchored thinking (R7/R8 + powered n=1000 eval): restricting candidate
  search to a behavior prior's support yields the first behavioral positive.
  ESTABLISHED (CI-clean): think beats no-thinking BC on success (+1.4pp
  [+0.4,+2.6]) and return (+1.74 [+1.20,+2.25]), and beats random candidate
  selection on return (+1.78; directed goal-reaching, 25.7 vs 37.2 steps).
  NOT RESOLVED: success over random candidate diversity (+0.5pp [-0.9,+1.8]).
  Binding constraints identified: BC anchor ceiling (60.9% acc, capped by the
  frozen planning-purposed context encoder) and value sharpness ->
  imagination-policy route as future work.
  (output/closed_loop_drone_game_v9_power/)

## 7. Qualitative: visible thinking traces

- Decoder-only fine-tune (motion-weighted) renders imagined futures on
  dm_control-style scenes: walker at advancing poses, candidate rows visibly
  differ (output/imagination_traces_armE_latest_v2dec/). Real-robot scenes
  limited by decoder capacity - latent metrics carry those claims.

## 8. Related work

- Dreamer v3/4 (policy-in-imagination, value bootstrapping, action-token
  dynamics), Dream-VLA/VLX (action-chunk prediction), offline MPC pitfalls /
  model exploitation, behavior-regularized offline RL, world-model evaluation.

## 9. Limitations and honest claims

- Reuse the handoff's safe/unsafe claim discipline. No real-drone claims.
- Selection margins thin on robot data; bridge acting transfer open; h16
  improved (1.7-2.8x) but not closed; DROID timing unidentifiable in-data
  (shift-1 cosine 0.91) - report as data property, not model failure.

## 10. Artifacts

- Scripts: eval_latent_imagination_decision_quality.py,
  eval_act_by_imagination.py, decode_imagination_traces.py,
  eval_gym_drone_game_act_by_imagination.py, collectors incl.
  collect_gym_drone_game_branch_dataset.py, relabel_rewards_return_to_go.py,
  finetune_tokenizer_decoder.py.
- Headline checkpoints: armE_seed2 final (offline), armE 170k / seed2 150k
  (robot windows), drone_game_v4_scoredrop (closed-loop best).
- Source docs: DECISION_QUALITY_AUDIT_RESULTS.md, ACT_BY_IMAGINATION_HARNESS.md.
