# Procgen Benchmark

## Why This Benchmark

Procgen is a well-known visual RL generalization benchmark:

```text
https://github.com/openai/procgen
```

It is not drone-specific, but it is useful for testing whether a visual representation improves sample efficiency and generalization from pixels.

## Current Scope

Implemented:

```text
Docker image for Python 3.10 Procgen runtime
random-policy smoke evaluator
RGB trace contact sheet
summary/report outputs
```

Not implemented yet:

```text
Kairos-feature policy
CNN baseline
world-model encoder baseline
training loop
matched-seed model comparison
```

## Build

```bash
cd /home/mkrzus/kairos-sensenova
./sensenova_drone_agent/scripts/build_procgen_benchmark_image.sh
```

## Run Smoke Test

```bash
./sensenova_drone_agent/scripts/run_procgen_benchmark.sh \
  --out-dir sensenova_drone_agent/output/procgen_coinrun_random_v2 \
  --env-name coinrun \
  --episodes 8 \
  --num-envs 4 \
  --max-steps 3000
```

Outputs:

```text
summary.json
report.md
episodes.jsonl
random_trace.png
```

## Paper Relevance

Procgen becomes paper-useful only after we compare:

- CNN from scratch.
- random frozen features.
- generic pretrained features.
- Kairos/Sensenova features.
- our lightweight drone-game world-model features, if compatible.

The first milestone is only dependency/API verification.

## First Result

Smoke output:

```text
output/procgen_coinrun_random_v2
```

Run:

```bash
./sensenova_drone_agent/scripts/run_procgen_benchmark.sh \
  --out-dir sensenova_drone_agent/output/procgen_coinrun_random_v2 \
  --env-name coinrun \
  --episodes 8 \
  --num-envs 4 \
  --max-steps 3000
```

Result:

```text
episodes_completed: 8
completed_requested: true
steps_executed: 1763
mean_return: 1.25
mean_length: 729.75
```

Interpretation:

```text
The Procgen Docker/runtime path works. This is a random-policy dependency
smoke test only, not a Kairos/Sensenova control result.
```
