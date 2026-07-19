# What the Right Data Looks Like

A sourcing spec for thinking-in-frames (action-conditioned world models +
act-time selection), distilled from the July 2026 campaign's verified
findings. Each requirement cites the experiment that established it.

## The one-line version

> Video of an embodied system where **actions are identifiable from the
> frames**, **coverage includes the states a policy actually visits**
> (especially failure-adjacent ones), and **labels are corrective
> (state → what an expert would do), not just outcome-tinted** — because
> outcome-labeled data from an improving agent teaches the world model
> that good behavior leads to death.

## Hard requirements (deal-breakers)

1. **Action identifiability** — the single most important measurable
   property. Compute shift-1 cosine similarity between consecutive raw
   action vectors: ≤ ~0.5 means timing is learnable; ≥ ~0.8 means no
   objective can ground *when* an action acts. (Expert game data: 0.33 —
   timing grounded to 27x error ratios. DROID: 0.91 — timing never
   learnable, a data property, not a model failure.) Prefer discrete or
   bang-bang control, or continuous control that isn't low-pass filtered
   into mush. Actions must be logged per-frame with a known frame/action
   alignment offset.

2. **Terminal outcomes must be *inside* the data windows.** Standard
   window enumeration silently excludes episode terminals from scored
   futures — the model then trains without ever seeing a consequence.
   Either episodes come with absorbing-state padding (repeat final frame +
   null action + zero reward) or the loader must add it. (Closed-loop
   round 4: the planner had never trained on any future containing a
   collision.)

3. **Deployment-distribution coverage.** Offline data from *other*
   policies does not certify act-time behavior — a policy searching over
   imagined futures reliably drives into states the data never covered
   (two-seed sign reversal; DAgger cycle-1 repaired exactly this and
   nothing else). Data must include the states the intended policy class
   visits, including near-failure states. If you can only get passive or
   expert-only data, budget for one round of on-policy collection.

## The poisoning constraint (subtle, campaign-specific discovery)

4. **Outcome labels must not anti-correlate with good behavior.** Data
   from a *competent-but-imperfect* agent is dominated by trajectories
   where goal-directed behavior ends in failure. Training dynamics on it
   teaches the imagination that good actions lead to doom — our exchange
   test localized the resulting worse-than-random selection entirely to
   the world model (8/8 cells, two seeds), with the value head exonerated.
   Screening heuristic: if P(failure | goal-directed prefix) in the
   dataset is much higher than P(failure | random prefix), the data will
   poison imagination training. Fixes in order of preference:
   - **corrective labels**: (visited state → expert/teacher action) pairs
     — this is what makes iteration ladder (classic DAgger's guarantee),
     now empirically confirmed in our stack: one round of expert-corrected
     labels cleared the campaign's all-time ceiling in both seeds and
     roughly doubled it by round 2 (Fig. 7 of the paper);
   - success-preserving reweighting (upweight the rare wins);
   - keep such data out of dynamics training entirely (policy-head only).

5. **Success-signal density.** Value/selection learning needs to see what
   winning looks like: as a floor, enough successful episodes that they
   are not statistically drowned (our 6% agent's data at ⅓ mix weight was
   already too dilute at 24 successes / 400 episodes). Aim ≥ 10–20% of
   episodes reaching the goal, or plan to reweight.

## Strongly helpful (not strictly required)

6. **Counterfactual branches**: multiple actions rolled out from the same
   state (sim snapshot/restore, or naturally repeated setups). Identifies
   action-effects directly. Note: helps offline objectives; behavioral
   value unproven in our runs — treat as identification aid, not a cure.
7. **Diverse action coverage in similar states** — beware datasets with a
   dominant modal action (the mean-action/argmax-collapse trap: 65%
   forward + never-yaw made discrete decoding degenerate).
8. **Episode lengths ≥ seq_len** (we use 24 frames: 8 context + 8 horizon
   + slack) with consistent dt between frames; stable camera/embodiment
   within a source (cross-embodiment transfer of *acting* never worked;
   only selection transferred).

## Format checklist (for our WMDataset layout)

- RGB frames (we run 128×128; higher is fine, we downscale).
- Per-transition: action (index or vector), reward, episode id, terminal
  flag. Known `action_frame_offset` (does action t produce frame t or
  t+1?).
- Episode-complete shards (no truncation mid-episode without a flag).
- If a teacher/expert exists (scripted policy, MPC oracle, human
  tele-operator willing to relabel): capture its action at visited states
  — this is the highest-value column in the whole spec (see #4).

## How much data (measured, not guessed)

Our confirmed ladder used surprisingly little: ~200 teacher episodes plus
~400 agent episodes with teacher labels (~25k labeled states total) per
round, with gains flattening after round 2. Two implications for sourcing:
- **A modest, well-targeted corrective set beats a large passive one.**
  Don't pay for volume before checking the properties above.
- **Data investment has a hard stop**: once the representation binds (our
  BC accuracy was encoder-capped at 61%), more/better data stops paying.
  If a pilot round of corrective data doesn't move behavior, suspect the
  encoder before the dataset.

## Tokenizer-domain match (for new visual domains)

Everything above concerns the planner's data. The frozen tokenizer has its
own requirement: the visual domain must be in (or near) its training
distribution. Cheap pre-check before committing to a source: encode →
decode a few hundred frames and inspect reconstructions; if the tokenizer
can't reconstruct the domain, either budget a decoder/tokenizer fine-tune
(our motion-weighted decoder recipe) or expect quantitative claims to be
latent-space-only. Teacher chunk labels: if the teacher can be queried in
sim, roll it forward one horizon from each visited state
(snapshot/restore) — chunk labels supervised our head better than
single-action labels would; from humans, short 8-step corrections at
failure points are the equivalent.

## Quick evaluation protocol for a candidate dataset

1. Shift-1 action cosine (< 0.5 good; > 0.8 reject for timing claims).
2. Fraction of episodes with terminal-in-window after padding (> 0 required).
3. P(failure | goal-directed prefix) vs P(failure | random prefix)
   (ratio ≫ 1 → poisoning risk, needs corrective labels or reweighting).
4. Success-episode fraction (≥ 10% comfortable; < 2% needs reweighting).
5. Modal-action share (< ~50%; higher → decode-collapse risk).
6. Does a teacher exist that can label visited states? (If yes, the
   dataset can support laddered improvement; if no, expect one-shot
   repair only.)
