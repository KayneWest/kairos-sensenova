# Act-By-Imagination Harness (Acting Leg)

Date: 2026-07-08

The rankfix campaign (see `DECISION_QUALITY_AUDIT_RESULTS.md`) established
that the latent imagination planner can propose action-causal futures and
select among them better than random, including on a held-out source. This
harness measures the remaining leg of the think-then-act loop: **do the
actions emitted from selected imagined futures beat actions emitted without
thinking?**

## Protocol

For each held-out context:

```text
1. sample K candidate plans (bank = true plans of other contexts)
2. propose futures, score with the trained scorer
3. select argmax-score candidate
4. action chunk = inverse_dynamics(ctx_h, selected_future, selected_plan)
5. compare to the LOGGED expert action chunk (masked MSE + cosine)
```

Controls and references per context:

```text
random candidate     inverse actions of a random plan's future
worst candidate      argmin-score candidate
zero-plan            actions from the zero-action plan's future
mean-action          dataset mean action chunk (blind prior floor)
repeat-last          last context action repeated over the horizon
oracle: true-plan    actions from the true plan's proposed future
                     (plan encodes the true actions - LEAKS, reference only)
oracle: real-future  inverse_dynamics on the REAL future latents
                     (inverse-dynamics ceiling given perfect foresight)
```

## Gates

```text
act_sel_beats_rand      selected action-MSE < random candidate (bootstrap CI)
act_improves_with_k     selected-minus-random improves from K=1 to K=64
act_sel_beats_zero      selected beats the zero-plan actions
act_beats_mean_action   selected beats the blind mean-action prior
held-out transfer       gates hold on bridge (never trained)
```

Where success/reward labels exist (SOAR), also report the positive-return
subset, where logged actions are demonstrably good.

## Known Off-Distribution Risk (measured, not assumed)

`inverse_dynamics` was trained only on (real future, true plan) pairs. Act
time feeds it (imagined future, candidate plan). If transfer fails here, the
fix for the next training run is adding inverse-on-imagined-futures (with
selected/candidate plans) to the training objective.

## Artifacts

```text
scripts/eval_act_by_imagination.py
scripts/experiments/launch_act_by_imagination.sh
output/act_by_imagination_<run>/summary.json + per-context JSONL
```

Checkpoints under test: `armE_seed2 final (210k)` (headline) and
`armD final (210k)` (control).

## Round 1 Verdict (2026-07-08): acting leg does NOT yet pass — mechanism found

Runs: `output/act_by_imagination_armE_seed2_final/`, `_armD_final/`
(256 contexts x 64 candidates, SOAR / expert / held-out bridge).

- Expert: selection helps actions (sel-minus-rand -0.057/-0.067, CI < 0,
  improves with K, 68-69% beats-random, cosine 0.20 vs 0.06 random) but does
  not beat the blind mean-action prior. 3/5 gates.
- Robot sources: selection does not transfer to action quality (SOAR +0.005,
  seed2 bridge +0.020; armD bridge -0.008 is the only robot CI pass). No run
  beats the mean-action prior anywhere.
- The inverse-dynamics CEILING is strong everywhere (oracle-on-real-future
  0.0045-0.15, ~= true-plan) - the head itself is capable.

Diagnosis (verified directly): the inverse head is largely a PLAN-DECODER.
On SOAR, shuffling the future latents changes its output MSE by nothing
(0.0153 vs 0.0150) while removing the plan blows it up 4.6x - it reads the
actions back out of the plan token, which encodes the true actions during
training. At act time candidate plans carry OTHER contexts' action content,
so the head emits the wrong actions regardless of how good the selected
imagined future is.

## Round 2 Fix (v4 "actfix", training now)

Two trainer additions (`--inverse-plan-dropout 0.5`,
`--inverse-imagined-weight 0.25`), continuation from armE_seed2 final on
GPU 1 (`latent_imagination_planner_all_data_v4_actfix`, 210k -> 240k):

1. Plan dropout on the inverse input: the head must decode actions from the
   future latents half the time.
2. Imagined-future inverse loss: `inverse(ctx, propose(ctx, plan(a)), 0) -> a`
   - supervises the exact act-time path (imagined future, no plan) with true
   actions, no act-time leakage.

Re-run this harness with `--inverse-plan-mode zero` (plan-free decoding) on
the v4 checkpoint. Also note: the mean-action prior is a strong floor on
action-matching metrics by construction (central predictions win MSE);
if selected actions beat random/zero/repeat-last but not the mean, consider
adding a closed-loop sim eval (gym drone game) for a behavioral verdict.

## Round 2 Verdict (v4, plan-free decoding): better actor, thinking still blocked

Run: `output/act_by_imagination_v4_actfix_zero/`. Diagnostic + harness agree:

- Plan-free absolute action quality improved a lot (SOAR 0.054 -> 0.031,
  cosine 0.48 -> 0.70; bridge 0.049 -> 0.021; expert 0.327 now BEATS the
  mean-action prior 0.378 and repeat-last everywhere).
- The act-time path works (imagined future decodes as well as real: 0.0298
  vs 0.0299 on SOAR).
- BUT selection changes nothing (sel-minus-rand ~ +/-0.0001 everywhere):
  with the plan zeroed the head decodes from ctx_h - it became a context-BC
  actor. Shuffling the future changes SOAR output by 0.0001. The
  future-reading pathway is still missing, so thinking cannot flow into
  emitted actions.

## Round 3 Fix (v5 "crossinv", training now)

`--inverse-cross-weight 0.5` continuation from v4 final (240k -> 270k),
`latent_imagination_planner_all_data_v5_crossinv`, GPU 1:

Cross-context imagined-inverse: hold ctx fixed, encode ANOTHER context's
action chunk a' into a plan, roll out the future it implies, and require
`inverse(ctx, propose(ctx, plan(a')), 0) -> a'`. The context cannot explain
the target; only the future can - this trains exactly the pathway that lets
candidate selection change emitted actions. Re-run the harness (zero mode)
on the v5 checkpoint; watch for sel-minus-rand going negative with K while
the absolute level holds near v4's.

## Closed-Loop Drone Game (behavioral verdict track)

Pipeline: `collect_gym_drone_game_dreamer4_dataset.py` (900 mixed-policy
episodes, 46k frames, 26.5k windows, outcomes 326 success / 282 collision /
292 timeout) -> tokenizer fine-tune (`drone_game_tokenizer_v1`) -> planner
with the full v5 recipe (`latent_imagination_planner_drone_game_v1`, 60k
steps, ~1h; offline metrics excellent: ret_corr +0.93, fid +0.94, zero 100x+,
tshift 8x) -> `eval_gym_drone_game_act_by_imagination.py` (matched-seed
episodes, paired CIs).

Closed-loop round 1 (inverse-head acting): ALL imagination policies crash in
~9 steps (heuristic ceiling 41.5% success). Verified cause: plan-free
discrete argmax decoding collapses to 3 of 9 actions (forward 65%, yaw
never) at 45% accuracy even on real futures - the discrete-argmax version of
the mean-action trap.

Closed-loop round 2 (action-native MPC: candidates are action chunks encoded
via encode_plan, winner's own actions executed - no inverse head): crashes
even faster (5.9 steps) while "winning" on short-horizon return. Diagnosis:
SCORER MYOPIA - the scorer regresses 8-step shaped return, and charging at
the goal maximizes progress reward with the collision beyond the credit
window. The classic reason Dreamer uses a bootstrapped value head.

Closed-loop round 3: rewards relabeled as full-episode discounted
return-to-go (`relabel_rewards_return_to_go.py`, gamma 0.99) and the planner
retrained with `--gamma 0` (episode-value scorer). Behavior IDENTICAL to
round 2 - which exposed round 4's finding.

Closed-loop round 4 (STRUCTURAL DATA BUG, affects every WMDataset run in
this repo): window enumeration can never place an episode's terminal
transition inside the scored future rows [ctx, ctx+h) - terminals only ever
appear in the final buffer rows. The planner had NEVER trained on a future
containing a collision, and pre-collision contexts were ~0.6% of windows.
Fix: absorbing-state padding in the collector (`--pad-terminal`, repeated
final frame + hover + 0 reward). Behavior still identical, exposing round 5.

Closed-loop round 5 (SCORER PLAN-SHORTCUT - the inverse-head disease again):
probes showed the proposer CORRECTLY imagines forward-into-tree as frozen
frames (crash prediction!), but the scorer routes through the plan token
(Q(ctx, plan)) and scores fwd +16.8 regardless of the imagined crash. Fix:
`--score-plan-dropout 0.5` + act-time `--score-plan-mode zero`. FIRST
behavioral movement: survival 5.9 -> 14.0 steps, first success (0.5%),
paired return gate passes CI (+4.0 [+3.3,+4.7]). Still 99% collision vs
heuristic 41.5% success.

Closed-loop round 6 (true counterfactual branch data):
`collect_gym_drone_game_branch_dataset.py` uses env snapshot/restore to roll
ALL 9 actions from the SAME state (4221 branches, concentrated at
blocked-front states) - the handoff's "contrastive thinking data", finally
real. Mixed at weight 2 with seq_len 16 windows (trunk ctx -> branch
future). Result: closed-loop REGRESSED (6-step crashes, negative return
delta vs random selection).

### Closed-loop track verdict (honest)

Six rounds, each eliminating a verified defect, and offline metrics improve
every time while closed-loop behavior barely moves. This is the known
signature of OFFLINE-TRAINED MPC EXPLOITING MODEL ERRORS under distribution
shift: argmax over imagined futures reliably finds states where the model is
confidently wrong, and act-time search compounds OOD drift that offline
gates cannot measure. This is precisely why Dreamer trains a POLICY inside
imagination (with a behavior prior) instead of doing act-time search.

Status: behavioral closure NOT achieved in the drone game; best result
round 5 (survival 2.4x random-candidate selection, paired return CI > 0,
0.5% success vs heuristic 41.5%). Every defect found is documented and the
counterfactual-branch infrastructure now exists.

### Recommended next directions (not yet run)

1. Imagination-policy route (Dreamer-proper): train a policy head with the
   PMPO machinery already in this repo against the drone planner's value,
   with a BC prior - replaces act-time argmax search with a trained policy,
   the standard cure for MPC model exploitation.
2. Online iteration (DAgger-style): collect episodes WITH the round-5 MPC
   agent, add them to training (its visited states enter the data
   distribution), retrain, repeat 2-3 cycles.
3. Uncertainty-penalized scoring: score = value - k * ensemble disagreement,
   directly attacking confident-but-wrong selection.

## Round 7: BC-anchored thinking — FIRST BEHAVIORAL POSITIVE

Direction 1 implemented lightweight: BC chunk head (frozen planner ctx
encoder + MLP, `train_drone_bc_chunk_head.py`, expert/eps-expert-only data)
supplies K=32 candidate chunks; imagination + plan-free value select among
BEHAVIOR-SUPPORT candidates only. `output/closed_loop_drone_game_v7_bcthink/`:

```text
act_bc_think   3.5% success  return -1.04  16.5 steps
act_bc         0.0%          return -3.53   6.3 steps  (argmax, no thinking)
act_bc_random  1.0%          return -2.95  20.4 steps
think-vs-bc      success +0.035 [+0.010,+0.065]  return +2.49 [+1.57,+3.63]
think-vs-random  success +0.025 [-0.000,+0.060]  return +1.91 [+0.80,+3.11]
```

Thinking beats plain BC on success AND return with CIs clear of zero — the
first closed-loop evidence that imagining and evaluating futures improves
real outcomes. The strict gate (think > random on success, CI > 0) misses by
a hair. Binding constraint moved to the BC anchor itself (60% action
accuracy; its argmax crashes in 6 steps).

## Round 8 + powered eval (n=1000): FINAL CLOSED-LOOP NUMBERS

Stronger anchor (1200 episodes, 1024-hidden head, 15k steps) plateaued at
60.9% action accuracy — the frozen planner ctx encoding, not data, caps BC
quality. Powered eval `output/closed_loop_drone_game_v9_power/`
(1000 matched seeds, act_bc_think / act_bc / act_bc_random):

```text
act_bc_think   2.8% success  return -0.42  25.7 steps
act_bc         1.4%          return -2.16  21.8 steps
act_bc_random  2.3%          return -2.20  37.2 steps

think vs bc      success +0.014 [+0.004,+0.026]   return +1.74 [+1.20,+2.25]
think vs random  success +0.005 [-0.009,+0.018]   return +1.78 [+1.20,+2.38]
```

Within this seed (CI-clean at n=1000): imagination-guided selection improves
SUCCESS RATE and RETURN over the deterministic no-thinking baseline, and
RETURN over random candidate selection (thinking reaches goals directedly:
25.7 vs 37.2 steps). NOT RESOLVED: a success-rate advantage over random
candidate diversity.

## Round 9b: REPEATABILITY SEED REVERSES THE RESULT

Full-stack repeat (planner SEED=20260888 -> BC head seed 20260889 -> n=1000
eval, fresh eval seeds; `closed_loop_drone_game_v10_power_seed2`):

```text
seed 2:  act_bc_think 0.9%  return -2.67  11.0 steps
         act_bc       2.2%  return -0.42  18.8 steps
         act_bc_random 3.7% return -0.59  30.5 steps
think-vs-bc     success -1.3pp [-2.3,-0.3]   return -2.26 [-2.73,-1.83]
think-vs-random success -2.8pp [-4.2,-1.5]   return -2.09 [-2.64,-1.61]
```

Sign REVERSAL with CIs clear of zero on both sides. Critically, seed 2's
OFFLINE metrics are equal or better (fid +0.78 vs +0.57, ret_corr +0.78 vs
+0.76, BC acc 60.7% vs 60.9%, tshift 8.1, zero 139) — offline gates do not
predict even the SIGN of closed-loop selection value. The seed-2 thinker
dies fastest (11 steps): its value head confidently prefers chunks that
crash.

### FINAL closed-loop verdict (supersedes round 8's)

The behavioral thinking-helps result is NOT seed-robust and is therefore
NOT claimable. The campaign's own lucky-seed rule applies to our own
headline. What survives, strengthened: offline decision-quality metrics —
even in-domain, near-ceiling ones — do not certify act-time behavior; the
gap is not noise but seed-level sign instability of the value head's
act-time preferences. Fixing it is the explicit target of the PMPO /
control-purposed-encoder campaign (policy trained in imagination with a
behavior prior, rather than argmax over a seed-fragile value head).

## Round 3 Verdict (v5): THINK-THEN-ACT LOOP CLOSED ON EXPERT

Run: `output/act_by_imagination_v5_crossinv_zero/`.

```text
                     expert           SOAR              bridge (held out)
gates                5/5 ALL PASS     4/5               1/5
sel-minus-rand (CI)  -0.0048 (<0)     -0.0002 (<0)      +0.0004
beats random         81% (v4: 53%)    76% (v4: 58%)     19%
vs mean-action       beats            ~ties (CI ~0)     loses
improves with K      monotone         yes               no
```

First end-to-end closure: on expert, actions decoded from the SELECTED
imagined future beat a random candidate's (CI), the zero-plan actions, the
blind mean-action prior, and repeat-last, improving monotonically with K.
SOAR passes everything except the mean-action CI (point estimate already
beats it). The result was reached by three verified, targeted fixes:
plan-decoder -> plan-free decode -> future-reading decode.

Honest boundaries:
- Selection margins are thin (~1.5% of action MSE on expert, ~0.7% on SOAR)
  vs the oracle gap (0.33 -> 0.14) - the 128-dim plan bottleneck limits how
  much action detail imagined futures can carry. Capacity (plan_dim /
  per-step readout width) is the magnitude lever.
- No transfer to held-out bridge: the future-reading decode is
  source-specific so far.
- These are offline action-matching proxies. The behavioral verdict needs a
  closed-loop sim eval (gym drone game) where selected-vs-random actions
  produce different returns.
