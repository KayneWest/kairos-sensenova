# Worklog — Thinking-in-Frames Campaign

Purpose: durable state so any session (or a fresh terminal) can resume.
Read order for full context:

```text
1. docs/ACTION_CONDITIONED_IMAGINATION_HANDOFF.md   (pre-campaign history)
2. docs/DECISION_QUALITY_AUDIT_RESULTS.md           (offline campaign, complete)
3. docs/ACT_BY_IMAGINATION_HARNESS.md               (acting leg, complete w/ final numbers)
4. docs/PAPER_SKELETON.md                           (paper outline, numbers wired)
5. this file                                        (live queue)
```

## Completed (July 2026 campaign)

- Decision-quality audit built + baseline diagnosed (scorer fidelity-inverted
  -0.97; training metric was circular). `eval_latent_imagination_decision_quality.py`.
- Offline loop fixed via 4 verified changes (rank loss, unit-norm plans,
  reward-detach, per-step plan conditioning). Timing phase transition
  (absolute-margin annealing). Two-seed repeatability. Headline offline ckpt:
  `latent_imagination_planner_all_data_v3_rankfix_armE_seed2/planner_ckpts/final.pt`.
- Offline acting closed (5/5 gates expert) after inverse plan-decoder fix
  chain (plan-dropout -> imagined-inverse -> cross-context). v5 robot ckpt:
  `latent_imagination_planner_all_data_v5_crossinv/planner_ckpts/final.pt`.
- Visible thinking traces (motion-weighted decoder fine-tune v2):
  `output/imagination_traces_armE_latest_v2dec/`.
- Closed-loop drone game, 9 rounds. Final powered result (n=1000,
  `output/closed_loop_drone_game_v9_power/`): BC-anchored thinking beats
  no-thinking BC on success (+1.4pp CI [+0.4,+2.6]) and return (+1.74), beats
  random selection on return (+1.78); success-over-random-diversity
  unresolved. Key artifacts: planner `latent_imagination_planner_drone_game_v4_scoredrop`,
  BC head `output/drone_bc_chunk_head_v2.pt`, harness
  `eval_gym_drone_game_act_by_imagination.py`.
- Structural findings preserved in memory + docs: WMDataset terminal
  exclusion (fix: `--pad-terminal` absorbing padding), plan-shortcut symmetry
  (fix: plan-dropout on any act-time head), counterfactual branch collector
  (`collect_gym_drone_game_branch_dataset.py`).

## Active queue (task list mirrors this)

1. [task 13, DONE] Paper draft: docs/PAPER_DRAFT.md (v1, full text, TODOs
   bracketed: arch figure, tables 1-2, trace figure inclusion, abstract
   trim). Figures: output/paper_figures/fig{1,2,3}*.png via
   scripts/make_paper_figures.py.
2. [task 14, DONE — RESULT REVERSED] Repeatability seed FAILED the sign
   test: seed 2 (closed_loop_drone_game_v10_power_seed2) has thinking
   CI-clean WORSE than both controls, despite equal-or-better offline
   metrics. Behavioral thinking-helps claim WITHDRAWN; paper reframed
   around the stronger finding (offline gates do not predict act-time
   selection sign; value-head act-time preferences are seed-fragile).
   Docs updated: harness round 9b, PAPER_DRAFT abstract/§6/§8.
3. [task 15, RUNNING overnight 2026-07-10, GPU 0] PMPO campaign, TWO full
   seeds (20260901/20260902) chained. Implementation done:
   - Stage A: planner trainer gained --bc-head-weight / --bc-encoder-grad
     (built-in BC chunk head, CE on one-hot targets, grads into ctx encoder
     = control-purposed representation). Runs:
     latent_imagination_planner_drone_game_v7_pmpo_s<seed>.
   - Stage B: scripts/train_drone_imagination_policy.py — PMPO in
     imagination (sample K=16 chunks from policy, imagine, plan-free value,
     sign-of-advantage weighting, KL to built-in BC prior; detached-sample
     log-probs). Out: output/drone_pmpo_policy_pmpo_s<seed>.pt (+ .metrics.jsonl;
     watch value_gain_vs_prior).
   - Stage C: eval gained --policy-head / --use-builtin-bc and
     act_policy / act_policy_think policies + gates
     (act_policy_vs_act_bc, act_policy_think_vs_*). n=1000 evals:
     closed_loop_drone_game_v11_pmpo_s<seed>.
   VERDICT RULE: only claim if the sign pattern repeats across BOTH seeds
   (today's reversal lesson).
4. [task 16, DONE — mixed] Robot v6 (score-dropout continuation):
   expert still 5/5 ALL PASS; bridge unchanged (0/5); SOAR REGRESSED
   (sel-minus-rand -0.0002 -> +0.0014, beats-random 76% -> 40%).
   Conclusion: late-continuation score-dropout destabilizes robot selection;
   keep all_data_v5_crossinv as the robot acting checkpoint; score-dropout
   belongs in from-scratch training (as in the drone stack).
   Results: output/act_by_imagination_v6_scoredrop/.
5. [task 17, QUEUED behind 4 on GPU 1] h16 curriculum: trainer gained
   --horizon-curriculum-max/-weight (supervised tail of an extended rollout,
   time embeddings clamp). Continuation v6->v7
   (latent_imagination_planner_all_data_v7_h16, 300k->330k), then
   decision-quality audit (decision_quality_audit_v7_h16). Gate: h16
   true_over_persistence <= ~1.0 on trained sources.

All three overnight chains are self-contained background scripts; results
land in the listed output dirs regardless of session state.

## 2026-07-11 morning: overheat restart + reconstruction (CURRENT STATE)

GPU overheated overnight; host restarted; orchestration scripts died but
docker-run containers/checkpoints survived. Reconstructed state:

- Task 16 (robot v6): COMPLETE before the crash — mixed result, see item 4
  above. Robot acting checkpoint remains all_data_v5_crossinv.
- PMPO seed 20260901: Stage A planner DONE (60k), Stage B policy DONE
  (drone_pmpo_policy_pmpo_s20260901.pt; final value_gain_vs_prior +0.055,
  kl 2.48 — policy improves imagined value over the BC prior). Stage C eval
  was cut mid-run.
- PMPO seed 20260902: Stage A planner DONE (60k); Stage B was cut.
- h16 (all_data_v7_h16): died at ~309k/330k, resumable (SAVE_EVERY=2000).

RE-ARMED (running now):
- GPU 0 chain: seed-20260902 Stage B policy -> BOTH seeds' Stage C n=1000
  evals (out: closed_loop_drone_game_v11_pmpo_s20260901 / _s20260902).
  Expected done ~3-4h. VERDICT RULE: claim only if the act_policy/
  act_policy_think vs act_bc sign pattern repeats across BOTH seeds.
- GPU 1 chain: h16 resume from latest.pt (309k -> 330k) -> decision-quality
  audit (out: decision_quality_audit_v7_h16). Gate: h16
  true_over_persistence <= ~1.0 on trained sources. Expected done ~5h.

If this session is gone: check `docker ps`; if chains died again, the two
re-arm commands are reproducible from this file — (a) train_drone_imagination_policy.py
for s20260902 then eval_gym_drone_game_act_by_imagination.py per item 3's
flags for both seeds; (b) relaunch launch_latent_imagination_planner.sh with
RUN_ID=all_data_v7_h16 RESUME_CKPT=.../latest.pt and the item-5 flags, then
launch_decision_quality_audit.sh. Consider checking GPU thermals
(nvidia-smi -q -d TEMPERATURE) before heavy dual-GPU load.

When PMPO evals land, write the two-seed verdict into
ACT_BY_IMAGINATION_HARNESS.md (+ paper §6/§9) and update memory
project-latent-planner-rankfix. When the h16 audit lands, record
h4/h8/h16 persistence ratios in DECISION_QUALITY_AUDIT_RESULTS.md.

## 2026-07-11 PAUSE (GPU shared with another machine; dual-load shutdowns)

ALL sda-* containers stopped; GPUs quiet (48/52 C). State at pause:

- PMPO: BOTH seeds' planners + policies are DONE
  (drone_pmpo_policy_pmpo_s2026090{1,2}.pt; seed1 value_gain_vs_prior
  +0.055). Both n=1000 evals CRASHED on a load bug (planner built without
  the new built-in bc_head) — FIXED in
  eval_gym_drone_game_act_by_imagination.py (enable_bc_head before load,
  compile-checked). NOT RERUN YET.
- h16: stopped mid-resume at ~309-315k/330k; resume via latest.pt.

### Resume commands when GPU is safe (run each as its own chain)

(1) PMPO evals only (~2h total, single GPU, run sequentially):
  for SEED in 20260901 20260902: docker run (dreamer image, one GPU)
    python3 scripts/eval_gym_drone_game_act_by_imagination.py \
      --ckpt output/latent_imagination_planner_drone_game_v7_pmpo_s$SEED/planner_ckpts/final.pt \
      --out-dir output/closed_loop_drone_game_v11_pmpo_s$SEED \
      --episodes 1000 --num-candidates 32 --replan-every 4 \
      --score-plan-mode zero --use-builtin-bc \
      --policy-head output/drone_pmpo_policy_pmpo_s$SEED.pt \
      --policies act_policy,act_policy_think,act_bc,act_bc_think,act_bc_random
(2) h16: relaunch launch_latent_imagination_planner.sh with
    RUN_ID=all_data_v7_h16 RESUME_CKPT=.../all_data_v7_h16/planner_ckpts/latest.pt
    + the item-5 flags (~4h), then launch_decision_quality_audit.sh
    RUN_ID=v7_h16 on soar/expert/bridge.

Consider single-GPU-at-a-time operation until the shutdown cause is found.

## 2026-07-11 THERMAL GUARD + RESTART UNDER GUARD

scripts/experiments/thermal_guard.sh: detached (setsid) watchdog, polls both
GPUs every 15s; ANY GPU >=80C or >=520W -> `docker pause` all sda-*
containers; resume at <=68C and <=420W; exit trap unpauses; only touches
sda-*. Log: output/thermal_guard.log. Stop: touch /tmp/sda_thermal_guard_stop.
Relaunch: setsid nohup bash scripts/experiments/thermal_guard.sh & .
First poll caught the OTHER machine's job at 79C/609W on GPU 0 — the
combined-load shutdown hypothesis is confirmed; our work now runs GPU 1
only, sequentially, with <=75C cool-gates between stages, until the other
job finishes.

## 2026-07-11 PMPO TWO-SEED VERDICT (task 15 CLOSED)

No cross-seed-consistent behavioral win (n=1000 x 2 seeds,
closed_loop_drone_game_v11_pmpo_s2026090{1,2}): the PMPO policy learns
SURVIVAL (72-76 steps, few collisions) but ~0% success; return-vs-BC flips
sign across seeds; act_policy_think adds nothing over act_policy.
Conclusion: lightweight offline PMPO (4k steps, offline contexts, KL to BC
prior) converges to safe-passive behavior; behavioral closure needs online
interaction or far longer imagination-RL — a future campaign, with all
infrastructure now in place (bc-encoder-grad planner flags,
train_drone_imagination_policy.py, act_policy* eval arms). This matches and
extends the paper's central finding; add one sentence to PAPER_DRAFT §9.

## 2026-07-19 GOOD-DATA VERDICT: TRUE DAGGER LADDERS (two seeds) — PAPER v1.6, PUSHED

run_expert_dagger.py: expert chunk labels via env snapshot->expert
rollout->restore at every visited state; imagination+judge FIXED (c1
stacks); only the BC head retrains per round on the aggregate. Results
(think / bc success, n=1000, v20_expert_dagger_s{1,2}_r{1,2,3}):
  s1: r1 6.2/8.7  r2 8.7/7.6  r3 8.3/9.8
  s2: r1 9.3/5.8  r2 11.1/7.5 r3 9.7/7.3
Both seeds clear the 6.1% all-time ceiling in ROUND 1 and stay above it
(11/12 post-round arm-points); best 11.1%; plateau ~10% vs 41.5% expert.
Think-vs-bc ordering stays seed/round-inconsistent (+-2pp) — selection
neither reliably helps nor hurts once the prior is strong (deliberation =
crutch for a weak policy; Dreamer's distill-then-drop endgame from the
other direction). Plateau + known encoder-cap (61% BC acc) locates the
next binding constraint in the frozen REPRESENTATION.
CONSTRAINT ORDERING (final): (1) action-causal imagination, (2)
corrective on-policy data, (3) representation capacity; judge & search
never binding. Paper v1.6: abstract, §6.3, §9, Fig 7. Data-sourcing spec
for the user: docs/RIGHT_DATA_SPEC.md.
Reboot note: box rebooted again under LIGHT load (2 GPUs <30C) — hardware
(PSU?) suspicion strengthened; systemd guard auto-recovered (first test).

## 2026-07-19 CAMPAIGN COMPLETE — PAPER v1.5 FINAL (all three workstreams two-seed)

PLAN_GRAD seed-2 (v18_plangrad_{good,bad}judge_seed2): 0.2%/0.1% success
— at BC's floor, BELOW random selection, judge-invariant. Two-seed
verdict: soft plan-manifold search never beats random (1.5% then 0.1-0.2%
vs argmax's 4.5-6.2%), seed-fragile even in its failure mode. Mechanism:
the §5 inverse-head thin link (argmax executes a discrete valid BC
candidate; soft methods must decode actions from an optimized future).

PAPER v1.5 (all formats regenerated, artifact same URL): §6.1 forensics
answer (imagination-dominance = data-poisoning signature; pre-DAgger
fragility distributed), §6.2 gains plan-grad two-seed + contrast-
diffusion knife-edge boundary, §9 claims both directions, §10 reduced to
the one genuinely open item (stabilizing contrast-diffusion training).

CAMPAIGN STATE: nothing running; GPUs idle; guard active (systemd).
Unpushed commits on thinking-in-frames-campaign: push + PR whenever the
user says. The campaign law survived every attempted refutation this
week: hard selection over an action-causal imagination is the only
winning configuration; failures localize to the imagination or the
action-decode path, never the judge or the search.

## 2026-07-19 CONTRAST-DIFFUSION VERDICT: NO STABLE ACTION-CAUSAL EQUILIBRIUM (5 configs) + PLAN_GRAD SEED-1

CONTRAST-DIFFUSION (c): five configurations, one thorough negative with a
mechanistic shape — likelihood + contrast on the flat-latent diffusion
class is a KNIFE-EDGE, no stable action-causal equilibrium found:
  v2  (FiLM only, no contrast):        stable, action-blind (ratio ~1.0)
  v3  (FiLM, 2-sided hinge w=1):       stable, action-blind (hinge
      saturated at margin for 30k steps — ignoring plan is cheaper)
  v3b (trunk-inject, 2-sided w=10):    BIT (wrong/true 3.0 at 4k; first
      plan-use ever) then eps-loss blowup -> NaN weights ~9k
  v3c (trunk, 1-sided hinge w=3):      plan_over_free 0.75 at 2-6k (first
      sample-level action-sensitivity) -> loss explosion ~5k -> NaN
  v3d (trunk, 1-sided w=1, lr 1e-4):   oscillated 0.56-1.54 around the
      equilibrium for 28k steps, never converged, then skip-cascade (111
      nonfinite steps). Killed.
POSITIVE ARTIFACT: the trunk-injection pathway CAN couple actions into
the generative prior (0.56-0.75 ratios, wrong/true 3.0) — it has just not
coexisted stably with likelihood training in any tested config. Paper
§6.2 gets this boundary. Candidate future fixes (not run): EMA weights,
contrast warmup schedule, x0-prediction parameterization, transformer
trunk over per-frame tokens.

PLAN_GRAD (a) seed-1 (v18_contrastdiff_{good,bad}judge): 1.5% success
BOTH judges — beats bc CI-clean (+1.0pp, +2.2 return) but TIES bc_random
(CIs span 0) and is judge-invariant. Soft plan-sphere optimization is
much weaker than hard argmax (5.3-6.2%); actions must decode through the
inverse head (the §5 thin link) instead of executing a discrete BC
candidate. Seed-2 evals RUNNING (v18_plangrad_{good,bad}judge_seed2).

## 2026-07-19 REVERSAL FORENSICS VERDICT: pre-DAgger fragility is DISTRIBUTED, not imagination-only

v19_revforensics grid (n=1000, common eval seed, think success vs own
controls bc/random): s1p-s1j 2.5% (mild +, replicates v9's 2.8%);
s1p-s2j 1.6% (null); s2p-s1j 2.1% (mild -); s2p-s2j 0.4% (strong -,
replicates v10 reversal). BOTH seed-2 components degrade thinking mildly
alone and compound when paired — unlike post-DAgger c2c where the
imagination carried the entire effect (8/8 cells). BOUNDARY FINDING:
imagination-dominance is specific to self-training poisoning; generic
seed-fragility in the low-signal pre-DAgger regime is distributed across
the imagination-judge system. Note confound handled: s2 bc head is
stronger (3.1% vs 1.4%), so s2 cells face higher control bars — but the
within-row judge effect and within-column proposer effect are both
present regardless. Paper §6.1 "open forensics" sentence gets this answer
in the v1.5 pass (batch with contrast-diffusion results).

## 2026-07-18 THREE WORKSTREAMS LAUNCHED (commit 40de81c): contrast-diffusion (c) + plan-grad (a) + reversal forensics (b)

(c) CONTRAST-DIFFUSION (GPU 1, chain `run_contrast_diffusion_campaign.sh`,
setsid; status output/contrastdiff_chain_status.log): trainer gained an
absolute-margin contrast hinge — at low-noise t, x0 reconstruction under
the TRUE plan must beat wrong-action plans (shuffle/zero/tshift/tperm/
treverse, 2 modes per step) by margin 0.05; wrong_over_true telemetry per
200 steps. Success criterion: sample plan_over_free < ~0.9 and closed-loop
diff_argmax/diff_guided leaving the bc_random band. Chain: train v3
seed1 (30k, ~3.5h) -> lambda sweep {0.25,1,4} -> n=1000 evals good+
inverted judges (v18_contrastdiff_{good,bad}judge) -> seed2 repeat.
(a) PLAN_GRAD arm rides in the same evals: 10 steps of gradient ascent on
the judge score THROUGH the GRU proposer, re-projected onto the unit-norm
plan sphere; plan-free inverse decode. Soft-vs-hard comparison on the
already-action-causal imagination.
(b) REVERSAL FORENSICS (GPU 0, 4 containers): the pre-DAgger v9/v10
seed-reversal stacks (drone_game_v4_scoredrop{,_seed2} + bc heads
v2{,_seed2}) exchanged proposer x judge, common eval seed 20260710,
n=1000 -> closed_loop_drone_game_v19_revforensics_{s1,s2}prop-{s1,s2}judge.
If success follows the proposer again, the original "value heads are
seed-fragile" claim re-attributes to imagination fragility (paper §6
two-seed-test paragraph would need a caveat -> §6.1 already flags it).

## 2026-07-17 DIFFUSION-THINK CAMPAIGN COMPLETE (TWO-SEED) — PAPER v1.4 FINAL

Seed-2 diffusion evals mirror seed 1 exactly: diff_prior/diff_guided 0%
success with 56-80% timeouts under both judges; diff_argmax judge-
invariant at bc_random level; proposer v2_seed2 plan-conditioning inert
(ratio 1.007) — the scene-prior finding is two-seed. All diffusion-thread
claims now two-seed-consistent
(closed_loop_drone_game_v16_diffthink_{good,bad}judge{,_seed2}).

PAPER v1.4 (all formats regenerated, artifact same URL): abstract carries
the exchange + guided-diffusion results; new contribution 6; §6 coda
mechanism CORRECTED (poison = learned dynamics, not value head); new §6.1
(exchange test, the 8-cell table) and §6.2 (guided generation: likelihood
is not enough; guidance can't rescue an action-blind prior; floor property
holds vacuously); §9 claims updated both directions; §10 next levers
(dynamics-aimed repair, contrast-trained generative proposer, plan-token
gradient ascent on the unit-norm sphere, exchange-decompose the v9/v10
reversal stacks); Fig 6 = exchange grid + diffusion floors.

CAMPAIGN-LEVEL TAKEAWAY (for talks): the campaign opened with a judge
that LOOKED broken (r=-0.97) where the real cause was training signals,
and closes with a judge that looked broken where the real cause was the
imagination. Both times the value head was the symptom. Thinking-in-frames
lives or dies on the action-causal quality of the frames.

## 2026-07-16 NIGHT: DECOMPOSITION COMPLETE (TWO-SEED, 8 CELLS) — THE IMAGINATION IS THE POISON, UNANIMOUSLY

gru_argmax (BC candidates -> proposer imagines -> judge picks, matched
seeds, n=1000/cell), success s1/s2, gate:
  c1 imagination x c1 judge:   5.3 / 6.2  PASS PASS
  c1 imagination x c2c judge:  6.2 / 4.5  PASS PASS
  c2c imagination x c1 judge:  0.0 / 0.5  fail fail
  c2c imagination x c2c judge: 0.0 / 0.7  fail fail
(v16_diffthink_{good,bad}judge_gruarg{,_seed2}, v17_decomp_*{,_seed2})

CONCLUSION (two-seed, decisive): performance follows the PROPOSER in
every cell; the judge is irrelevant in every cell. The c2c "inversion"
never lived in the value head — the c2c scorer ranks c1-imagined futures
as well as the c1 scorer does. Failure-concentrated self-data poisoned
the IMAGINATION: and since worse-than-random selection (v15) requires
anti-correlation, the c2c proposer is SYSTEMATICALLY anti-goal — trained
on 94% "goal-directed flight -> crash" episodes, it imagines crash-shaped
futures precisely for good actions, so even an honest judge picks against
the goal. Corrupted dreams, honest judge.

REINTERPRETATIONS REQUIRED:
- Paper §6 cycle-2 coda mechanism sentence ("value head learns
  approaching-goal predicts death") -> the RTG-labeled data corrupts the
  learned dynamics/imagination, not (primarily) the value head.
- The "amplifier" finding re-aims: deliberation amplifies the quality of
  the IMAGINATION (0.0-6.2% span tracks the proposer, not the judge).
- Open (future forensics): decompose the v9/v10 seed-reversal stacks the
  same way — the original "value heads are seed-fragile" claim may also
  localize to imagination differences.
- DAgger repair re-read: cycle-1's win = on-policy data repairing the
  IMAGINATION's coverage; cycle-2's regression = the same channel poisoned.

Still running: main chain seed-2 diffusion evals (GPU 1). Full writeup +
paper pass when they land.

## 2026-07-16 EVENING: SEED-1 RESULTS + A MAJOR SURPRISE — THE POISON IS THE IMAGINATION, NOT THE JUDGE

Proposer v1 bug found+fixed: conditioning only on lossy ctx_h and diffusing
ABSOLUTE latents -> eps-loss plateau 0.55, samples 1000x worse than
persistence. v2 = persistence-delta parameterization + explicit last_z
conditioning + per-dim norm (mirrors the GRU's residual design): sample MSE
0.0048 at 600 steps, beats persistence (commit 8cd75e1).

FINDING 1 (scene-prior disease, generative form): v2 trains to
persistence-level quality but plan-conditioning stays INERT (plan/free MSE
ratio ~1.0 for all 30k steps). Pure likelihood recapitulates
scene-priors-not-action-causality. diff_argmax is therefore judge-noise
selection; added gru_argmax (action-conditioned GRU proposer + swappable
judge) as the honest amplifier arm (commit 330664c).

FINDING 2 (seed1, n=1000, good judge): all diffusion arms fail even with
the good judge — bc 0.5 / bc_random 1.9 / diff_prior 0.9 / diff_argmax 1.6
(= bc_random, CI spans 0) / diff_guided 0.0% with 70% timeouts. Guidance
AMPLIFIES PASSIVITY: judge gradient prefers "nothing happens" futures,
plan-free inverse decodes near-persistence as hover (absorbing-state
training pairs frozen frames with hover). Lambda sweep flat 0.25-16.
Under the bad judge diff_guided sits at the prior floor (0.3% vs 0.9%,
CI spans 0) — floor property holds but vacuously. VERDICT SHAPING UP:
guidance has no leverage on an action-blind prior; "soft thinking" needs
an action-causal generative manifold.

FINDING 3 (THE SURPRISE, seed1): gru_argmax with the C2C "inverted" JUDGE
scores 6.2% success, gate PASS (= c1's 6.1%); with the c1 judge 5.3% PASS
(cross-validates v12 via the new harness). closed_loop_drone_game_v16_
diffthink_{good,bad}judge_gruarg. THE C2C SCORER IS FINE when judging C1-
proposer futures -> cycle-2c's worse-than-random selection does NOT live
in the value head. Reinterpretation pending confirmation: failure-
concentrated self-data poisoned the c2c PROPOSER (imagination maps good
actions to crash-shaped futures; an honest scorer then picks against the
goal). Paper §6 coda's mechanism sentence will need revision.
DECOMPOSITION RUNNING (GPU 0, closed_loop_drone_game_v17_decomp_
c2cprop-{good,bad}judge): c2c proposer+encoder+BC-head under each judge.
If c2cprop-goodjudge collapses -> imagination is the poison, confirmed.

Chain continues on GPU 1: proposer v2 seed2 training, then seed-2 evals.

## 2026-07-16 DIFFUSION-THINK CAMPAIGN RUNNING (GPU 1): does guided generation bound the amplifier?

USER IDEA -> CAMPAIGN: treat latent->image as diffusion; "force z down a
thinking trajectory before releasing it" = value-GUIDED denoising (soft,
prior-anchored optimization) vs our argmax-over-K (hard optimizer that the
amplifier finding indicts). Unique asset: certified-good judge (c1) +
certified-inverted judge (c2c) from the cycle-2 arc.

NEW CODE (commit pending this entry):
- scripts/train_latent_diffusion_proposer.py — eps-pred FiLM-MLP (60M) over
  flattened future chunk (8x512=4096), cosine DDPM T=1000, DDIM 30,
  conditioned on FROZEN c1 planner ctx_h + plan (dropout 0.5 -> plan-free
  sampling works); ddim_sample() supports value guidance (grad of judge
  score wrt x_t through the x0 estimate). Trains on drone_v4_dagger
  manifest (the c1 winning mix). ~2.4 steps/s -> 30k ~= 3.5h.
- scripts/eval_gym_drone_game_diffusion_think.py — closed-loop, matched
  seeds, policies: bc / bc_random / diff_prior (lambda=0 control) /
  diff_argmax (K BC candidates, plan-conditioned samples, judge argmax) /
  diff_guided (ONE plan-free sample steered by judge gradient, plan-free
  inverse decode). Proposer stack fixed; JUDGE swappable (--judge-ckpt).
  Paired bootstrap gates incl. diff_{argmax,guided}_wins.
- scripts/experiments/run_diffusion_think_campaign.sh — chain: train
  proposer s1 -> lambda sweep {0.25,1,4,16} n=200 on GOOD judge -> full
  n=1000 evals under good (v16_diffthink_goodjudge) AND inverted
  (v16_diffthink_badjudge) judges -> repeat full stack seed2 (c1_seed2 /
  c2c_seed2 judges, eval seed 20270300).
PREDICTION: argmax flips sign with the judge (known); guided degrades
toward the prior under the inverted judge (floor rises from
worse-than-random to ~BC) while retaining a win under the good judge.
Status: output/diffusion_think_chain_status.log. ETA ~12h.
After results: update paper (new §; likely paper-2 opener) + this log.

## 2026-07-15 CYCLE-2 ARC COMPLETE: REPAIR, NOT LADDER (all cells two-seed; paper v1.3 final)

CYCLE 2c VERDICT (closed_loop_drone_game_v15_dagger_c2_only{,_seed2}):
replacement (base w2 + c2 w1, no c1 — exact c1 recipe shape, fresher data)
is the STRONGEST failure, both seeds: think 0.0%/0.5% success, returns
-8.69/-2.60, collision 95/99%; selection WORSE THAN RANDOM CI-clean on
success in both (seed1 -1.7pp + return -6.18; seed2 -1.4pp [-2.4,-0.5]).
The §3 scorer inversion re-emerges from DATA (not objective).

FINAL ARC (think success s1/s2, strict gate):
  c1  base+c1 (1/3):      6.1 / 5.7   PASS PASS   <- the only stable recipe
  c2  +c2 (1/2):          0.3 / 2.2   fail fail
  c2b +c2 (1/3):          2.9 / 0.7   fail fail
  c2c base+c2 only (1/3): 0.0 / 0.5   fail fail (selection < random, both)
MECHANISM: improved agent's episodes = long goal-directed flights ending
in collision; RTG labels attach negative outcomes to goal-approaching
futures -> value head learns "approaching goal predicts death." As agents
improve, failures concentrate along best trajectories; outcome-labeled
self-imitation becomes anti-goal evidence. Classic DAgger avoids this via
expert corrections on visited states.
KEY INSIGHT (paper-worthy): thinking is an AMPLIFIER, not a safety net —
across judges, random selection stays 1.3-2.2% while think-selection spans
0.0-6.1%: argmax over K futures multiplies judge quality in either sign.

Paper v1.3 FINAL: abstract boundary sentence, §6 coda (two-seed
throughout), §9 disclaims iterated self-improvement, §10 levers
(success-preserving replay / expert relabeling / online value / uncertainty
scoring), Fig 5 = the arc. All formats regenerated (HTML/PDF/artifact).
Next campaigns (open): success-upweighted replay test; online value
learning; uncertainty-penalized scoring.

## 2026-07-15 CYCLE 2b VERDICT: REBALANCE DID NOT RESTORE THE WIN — CYCLE 2c (REPLACEMENT TEST) RAN

c2b (same data as c2, dagger fraction restored to 1/3) two-seed verdict:
  seed1 (closed_loop_drone_game_v14_dagger_c2_rebal): think 2.9% / -0.32 vs
    bc 0.3% / -6.56 vs random 2.1% / -1.99 — beats BC CI-clean (success
    +2.6pp [+1.5,+3.8], return +6.24) but ties random on success
    (+0.8pp CI spans 0) -> strict gate FAILS.
  seed2 (..._seed2): think 0.7% / -2.57 vs bc 0.7% vs random 1.3% — NULL
    on success (ties bc exactly, -0.6pp vs random spans 0); only returns
    positive. Gate FAILS.
Partial-dilution conclusion: fraction explains some of c2's collapse
(seed1 0.3->2.9) but NO mix containing c2 data passes the gate in any
seed. Running arc: c1 6.1/5.7 gate PASS both; c2 0.3/2.2 FAIL; c2b
2.9/0.7 FAIL.

CYCLE 2c RUNNING (launched ~09:40, `run_dagger_cycle2c.sh`): the
replacement test — exact c1 recipe shape with newer data, base w2 +
dagger_c2_rtg w1 ONLY (no c1 data). Manifest drone_v7_dagger2_only;
planners drone_game_v11_dagger_c2_only{,_seed2}; BC heads
drone_bc_chunk_head_dagger_c2_only{,_seed2}.pt; evals
closed_loop_drone_game_v15_dagger_c2_only{,_seed2}. Wins => compounding =
rolling replacement of self-data. Fails => c2 data itself is the problem;
cycle-1 was a one-time distributional repair. Status:
output/dagger_c2c_chain_status.log.

## 2026-07-15 DAGGER CYCLE 2 VERDICT: REGRESSION IN BOTH SEEDS — CYCLE 2b (REBALANCE TEST) RAN

Cycle-2 chain completed 01:52. TWO-SEED VERDICT: naive DAgger accumulation
does NOT compound — it destroyed the cycle-1 win.
  seed1 (closed_loop_drone_game_v13_dagger_c2):     think 0.3% success /
    -3.39 return, collision 99.1% — CI-clean WORSE than act_bc (success
    -1.1pp [-2.0,-0.3]) and act_bc_random (-1.8pp, return -0.92) on
    success; return-vs-bc still +1.79 CI-clean (charges goal, then crashes).
  seed2 (v13_dagger_c2_seed2): think 2.2% / -1.17 vs bc 1.7% / -1.18 vs
    random 1.7% / -2.48 — NULL: all success CIs span zero; only
    return-vs-random positive (+1.31).
  bc_thinking_wins=false in both. Compare cycle 1: 6.1% / 5.7%, wins both.
Not a chain bug: BC head acc 61% (= c1), episodes ~36 steps, matched eval
seeds. The single changed variable: training mix DAgger fraction went 1/3
(c1) -> 1/2 (c2), both DAgger sets ~94% collisions.

HYPOTHESIS: failure-data dilution of the success signal. TEST RUNNING
(cycle 2b, `scripts/experiments/run_dagger_cycle2b.sh`, launched 02:15,
GPU 1, no new collection): retrain on the SAME data with dagger fraction
restored to 1/3 — manifest drone_v6_dagger2_rebal (base w4 + c1 w1 + c2
w1) -> planner drone_game_v10_dagger_c2_rebal{,_seed2} -> BC heads
drone_bc_chunk_head_dagger_c2_rebal{,_seed2}.pt -> evals
closed_loop_drone_game_v14_dagger_c2_rebal{,_seed2} (matched eval seeds).
If think returns to ~6%: dilution confirmed, and the compounding recipe is
fraction-controlled replay. If not: the cycle-1 win is fragile to
retraining with any new data — sharper negative. Status:
output/dagger_c2b_chain_status.log. Chain adds thermal_gate() between
stages (settle to <=75C / <=600W, max 15min wait).

PAPER IMPACT (apply after c2b lands, one pass): §6 DAgger paragraph keeps
the c1 two-seed claim (still true) but gains a coda — a second naive
accumulation round regresses (0.3%/2.2%, gate fails both seeds); §9 add
"we do not claim: iterated self-improvement from naive data accumulation";
Fig 2 possibly gains the c2 panels or a cycle-curve.

## 2026-07-14 DAGGER CYCLE 2 RUNNING (GPU 1) + THERMAL GUARD v3 UNDER SYSTEMD

GPUs freed (other research done for now); chain relaunched 19:44 CT and
correctly SKIPPED collect/relabel (cycle-2 data already on disk) — planner
seed1 drone_game_v9_dagger_c2 training (~9 steps/s, 60k ~1.8h), then BC ->
eval v13_dagger_c2 -> seed2 repeat. Monitor live on
output/dagger_c2_chain_status.log. User: run as many cycles as we want —
if cycle 2 compounds, cycle 3 = same script with c2->c3 bumped (and add a
thermal_gate() between stages, per optimal-z pattern; do NOT edit
run_dagger_cycle2.sh while it is executing — bash reads scripts lazily).

THERMAL GUARD v3 (commit 2e96f95), lessons transplanted from the user's
~/optimal-z agent's guard ON THIS SAME BOX: (1) sustained rules — 10-min
avg max-temp >=72C forces 180s cooldown (heat-soak: this box held 79C/588W
sustained without crossing any instant trip), 5-min avg total power >=850W
forces 90s cooldown; instant trips unchanged (80C / 950W sum). (2)
PERSISTENCE: now runs as user systemd unit sda-thermal-guard.service
(Restart=on-failure, loginctl linger) — survives reboots/session
teardowns; the setsid guard died on all three reboots to date. Manage via
`systemctl --user {status,stop,restart} sda-thermal-guard`. Caveat: paused
containers keep VRAM (guard sheds heat/power, never frees the card).
Neighbor project runs its own SIGSTOP-based guard for its processes.

## 2026-07-12 DAGGER CYCLE 2 — COLLECTION DONE, TRAINING BLOCKED (machine reboot; waiting on shared GPU)

CHAIN: `scripts/experiments/run_dagger_cycle2.sh` — the full 2-seed cycle-2
recipe (exact cycle-1 chain with c1->c2 bumped), setsid-detached, RESUMABLE:
every step skips itself if its artifact already exists. Status file:
`output/dagger_c2_chain_status.log`.

DONE (survived on disk):
- [1/9] Collection: 400 episodes with the cycle-1 winner agent (planner
  drone_game_v8_dagger_c1 + drone_bc_chunk_head_dagger_c1.pt, collect seed
  20270200) -> `data/gym_drone_game_dreamer4/dagger_c2` (19,635 frames).
  24/400 success (6.0% — matches the cycle-1 eval rate), 376 collisions,
  ~49 frames/ep vs ~40 in cycle-1 collection (agent survives longer).
  Cycle-2 training now sees ~2x the success episodes cycle 1 had (24 vs 13).
- [2/9] RTG relabel -> `dagger_c2_rtg`. [3/9] manifest
  `drone_v5_dagger2_manifest.json` (base w2 + dagger_c1_rtg w1 + dagger_c2_rtg w1).

BLOCKED: [4/9] planner seed1 died at init 08:54:02 with
cudaErrorDevicesUnavailable (GPU became busy — the neighbor project took the
devices), and the MACHINE REBOOTED ~11:52. Diagnostic note: our GPU load had
been dead since 08:54, so this shutdown happened with ONLY the neighbor
research job running — the instability is not (only) our dual-load.
USER CALL: wait for the other research to finish before resuming GPU work.

RESUME (one command, once GPUs are free; guard must be alive — check
`cat /tmp/sda_thermal_guard.pid`, relaunch with
`setsid nohup bash sensenova_drone_agent/scripts/experiments/thermal_guard.sh &`):

  setsid nohup bash sensenova_drone_agent/scripts/experiments/run_dagger_cycle2.sh \
    > sensenova_drone_agent/output/dagger_c2_chain.log 2>&1 &

Chain resumes at [4/9]: planner drone_game_v9_dagger_c2 (SEED=20260710, 60k)
-> BC head drone_bc_chunk_head_dagger_c2.pt (drone_bc_v2 data, 15k, hidden
1024) -> n=1000 eval closed_loop_drone_game_v13_dagger_c2 (eval seed
20260710, matched to cycle 1) -> seed2 repeat (SEED=20260711 / BC 20260712 /
eval 20270300 -> closed_loop_drone_game_v13_dagger_c2_seed2). Question: do
gains compound from ~6% toward the 41.5% heuristic ceiling?

Thermal guard UPDATED (and relaunched post-reboot): power trigger is now
TOTAL draw across both GPUs (pause >=950W sum, resume <=780W sum,
per-dimension hysteresis) instead of 520W per-GPU max — the old rule
thrashed on the neighbor job's solo ~600W bursts, which are safe alone; the
shutdown risk it models is combined load. Temp rule unchanged (80C/68C).

PAPER TRACK COMPLETE (2026-07-12, no GPU): `make_paper_figures.py` now
renders 5 figures to `output/paper_figures/`: fig0_architecture
(perceive/think/act lanes + the four fixes + plan-free act-time heads),
fig1 (unchanged), fig2 REBUILT as the four-panel arc (pre-DAgger seed-1
win -> seed-2 reversal -> post-DAgger consistency in both seeds — the
whole paper in one figure), fig3 (unchanged), fig4_trace_grid (expert
ctx 3: selected 0.0034 ~ true 0.0028, random 5.8x, zero-plan 16x worse;
walker visibly action-conditioned). PAPER_DRAFT.md bumped to v1.2: all
figure references wired (Fig 0/2/4), contribution 5 now carries the DAgger
ending, Table 1b fully populated from the audit JSONs (no n/r cells left),
§11 artifact list consolidated inline. Remaining paper TODOs: none
structural — only [verify] citation markers in §8 and final polish.

## 2026-07-11 TRACK 2: DAGGER CYCLE 1 — STRICT GATE PASSES (first time)

collect_dagger_episodes.py ran the act_bc_think agent for 400 episodes
(387 collisions — its own failures became training data) -> RTG relabel ->
retrain planner+BC on base(w2)+dagger(w1) mix
(drone_v4_dagger_manifest.json) -> n=1000 eval
(closed_loop_drone_game_v12_dagger_c1):

  act_bc_think 6.1% success, return +1.54 (FIRST positive-return agent)
  vs act_bc: success +5.6pp [+4.2,+7.1]; return +6.92 CI-clean
  vs random: success +4.7pp [+2.9,+6.4]; return +3.99 CI-clean
  bc_thinking_wins = True

2026-07-12: SEED-2 REPEAT CONFIRMS (closed_loop_drone_game_v12_dagger_c1_seed2):
think 5.7% / return +1.60, bc_thinking_wins=True, all comparisons CI-clean,
magnitudes matching seed 1. CLAIMABLE under the two-seed rule. Paper
abstract/§6/§9 updated with the constructive conclusion (failure is
distributional; one DAgger round suffices in this domain). Next optional:
DAgger cycle 2 with the improved agent (expect further gains).

Distribution-gap hypothesis CONFIRMED behaviorally. Claimability pended the
independent-training-seed repeat (RUNNING, GPU 0: runs
drone_game_v8_dagger_c1_seed2 / drone_bc_chunk_head_dagger_c1_seed2.pt /
closed_loop_drone_game_v12_dagger_c1_seed2; same DAgger data, fresh
planner/BC/eval seeds). If it repeats: update paper §6/§10 (DAgger converts
the reversal into a consistent win = paper #2 opening result or a §6 coda),
memory, and consider cycle 2 (collect with the improved agent).
Paper draft v1.1 committed (e2ba702); campaign commit e224bba, branch
thinking-in-frames-campaign, NOT pushed.

## 2026-07-11 H16 VERDICT (task 17 CLOSED) — WORKSTREAM RUN QUEUE EMPTY

Horizon-tail curriculum did NOT close h16 (audit decision_quality_audit_v7_h16):
expert 3.12 / SOAR 6.98 at h16 (slightly worse than seed2-210k baseline),
bridge improved 11.8 -> 6.7 (h8 0.86, below persistence held-out for the
first time). Confound: ran on the v6 score-dropout lineage (itself a known
regression). Clean retest would branch --horizon-curriculum-max 16 from
armE_seed2 final. h16 documented as open in PAPER_DRAFT §8.

ALL RUNS IN THIS WORKSTREAM ARE NOW COMPLETE. Remaining: none (docs current).
Future campaigns (new workstream): online imagination-RL for behavioral
closure; clean h16 retest; source-balanced rank loss; submission polish.

Historical note (superseded sections below): the pre-guard pause plan and
resume commands are retained for reference.

Remaining queue relaunched under guard (single chain, GPU 1):
PMPO eval s20260901 -> s20260902 (closed_loop_drone_game_v11_pmpo_s*) ->
h16 resume 309k->330k -> decision_quality_audit_v7_h16. Then only doc
wrap-up remains (PMPO two-seed verdict into harness doc + paper §6/§9 +
memory; h16 ratios into audit doc).

## Environment notes

- Everything runs in docker image `sensenova_drone_agent-dreamer:local`,
  repo mounted at /workspace, GPUs via `--gpus device=N` (2x RTX 5090).
- Launchers: scripts/experiments/launch_latent_imagination_planner.sh
  (env-var driven, see payload for flags), launch_decision_quality_audit.sh,
  launch_act_by_imagination.sh.
- Drone domain trains ~17 steps/s (60k steps ~= 1h). Robot mix ~0.75 s/step.
- Do not run destructive git commands; worktree is intentionally dirty.
