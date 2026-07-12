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
