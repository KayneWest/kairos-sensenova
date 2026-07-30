# Grounded World-Model Control

## Objective

The target loop is:

```text
real Gazebo camera frame I_t
    -> world-model state or surrogate state h_t
    -> movement proposal: forward / strafe / yaw / vertical / hover
    -> safety-filtered DroneCommand
    -> PX4 SITL actuation
    -> next real Gazebo camera frame I_t+1
```

The next state must come from the simulator camera after actuation. Generated
Kairos frames are hypotheses only; they are not stored as real memory and do
not become the next observation.

## Runtime Mode

Use:

```yaml
runtime:
  mode: "grounded_world_model"
```

The runtime router calls `GroundedWorldModelMovementPlanner`, which expects the
configured world-model side to expose:

```python
propose_movement(world_state, memory, goal, episode_step_dir=None)
```

That method returns a `GroundedMovementProposal`, which is converted into a
`PolicyOutput` and then filtered by `SafetyShield` in the normal closed-loop
agent.

## Current Concrete Backend

`BCGroundedWorldModelAdapter` is the current testable backend. It wraps the
trained BC visual policy and exposes it through the grounded world-model
interface.

This is a surrogate, not native Kairos latent control:

```text
real frame -> BC visual policy -> movement proposal
```

It exists so the full grounded action loop can be exercised now. Later, this
class should be replaced with a Python Kairos backend that returns:

```text
real frame + memory -> Kairos h_t -> action head -> movement proposal
```

## Intended Future Backend

`PythonKairosAdapter` should eventually implement:

```python
encode_observation(...)
encode_observation_and_memory(...)
propose_movement(...)
```

The key replacement point is `propose_movement`: once Kairos exposes a usable
decision state or movement head, the drone loop should not need to change.

## Telemetry Contract

Grounded world-model steps log:

```json
{
  "decision_rule": "world_model.propose_movement(real_observation_state)",
  "generated_rollouts_used_as_state": false,
  "candidate_rollouts": [
    {
      "action_sequence": ["strafe_left"],
      "world_model_proposal": {}
    }
  ]
}
```

This is the guardrail that keeps the system grounded: action decisions may come
from the world-model side, but state is refreshed only by real Gazebo frames.
