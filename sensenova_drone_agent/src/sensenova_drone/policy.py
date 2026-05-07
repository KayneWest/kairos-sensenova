from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand, discrete_to_command
from sensenova_drone.memory import RealObservationMemory
from sensenova_drone.scoring import GoalSpec
from sensenova_drone.world_state import WorldState


@dataclass
class PolicyOutput:
    action: DiscreteDroneAction
    command: DroneCommand
    action_logits: Any | None = None
    confidence: float | None = None
    value: float | None = None
    safety_risk: float | None = None
    metadata: dict | None = None

    @property
    def proposed_command(self) -> DroneCommand:
        return self.command


if nn is not None:
    class KairosPolicyHead(nn.Module):
        """
        Small policy head on top of a Kairos/Sensenova decision state h_t.
        """

        def __init__(
            self,
            input_dim: int,
            num_actions: int,
            hidden_dim: int = 512,
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.action_head = nn.Linear(hidden_dim, num_actions)
            self.value_head = nn.Linear(hidden_dim, 1)
            self.safety_head = nn.Linear(hidden_dim, 1)

        def forward(self, h_t):
            x = self.net(h_t)
            return {
                "action_logits": self.action_head(x),
                "value": self.value_head(x),
                "safety_risk": torch.sigmoid(self.safety_head(x)),
            }
else:
    class KairosPolicyHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("torch is required to instantiate KairosPolicyHead.")


class PolicyRuntimePlanner:
    """
    Runtime policy mode.

    This does not generate future videos.
    It uses:

        h_t = world_model.encode_observation_and_memory(...)
        action_logits = policy_head(h_t)
        a_t = argmax(action_logits)
    """

    def __init__(
        self,
        world_model,
        policy_head: KairosPolicyHead | None,
        action_cfg: dict,
        device: str = "cuda",
    ):
        self.world_model = world_model
        self.policy_head = policy_head
        self.action_cfg = action_cfg
        self.device = device

    def plan(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        goal: GoalSpec,
        episode_step_dir: str | None = None,
    ) -> PolicyOutput:
        _ = episode_step_dir
        h_t = self.world_model.encode_observation_and_memory(
            world_state=world_state,
            memory=memory,
            goal=goal,
        )

        if h_t is None:
            raise RuntimeError(
                "Policy mode requested, but world_model did not return h_t. "
                "Use MPC mode or implement PythonKairosAdapter.encode_observation_and_memory()."
            )

        if self.policy_head is None:
            raise RuntimeError("Policy mode requested, but no policy_head is loaded.")

        if torch is None:
            raise RuntimeError("Policy mode requested, but torch is not installed.")

        if not hasattr(h_t, "to"):
            raise RuntimeError("Policy mode requires h_t to be a torch-compatible tensor-like object.")

        tensor = h_t.to(self.device)
        if getattr(tensor, "dim", lambda: 0)() == 1:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = self.policy_head(tensor)
            logits = outputs["action_logits"]
            probs = torch.softmax(logits, dim=-1)
            action_idx = int(torch.argmax(probs, dim=-1).item())
            confidence = float(torch.max(probs).item())

        actions = list(DiscreteDroneAction)
        action = actions[action_idx]
        command = discrete_to_command(action, self.action_cfg)

        return PolicyOutput(
            action=action,
            command=command,
            action_logits=logits.detach().cpu(),
            confidence=confidence,
            value=float(outputs["value"].item()),
            safety_risk=float(outputs["safety_risk"].item()),
            metadata={
                "mode": "policy_head",
                "note": "No future generation used in this mode.",
            },
        )


class RuntimeModePlanner:
    """
    Routes runtime requests between MPC, policy, and hybrid modes.
    """

    def __init__(
        self,
        mpc_planner,
        cfg: dict,
        policy_planner: PolicyRuntimePlanner | None = None,
        grounded_world_model_planner=None,
    ):
        self.mpc_planner = mpc_planner
        self.policy_planner = policy_planner
        self.grounded_world_model_planner = grounded_world_model_planner
        self.cfg = cfg

    def plan(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        goal: GoalSpec,
        episode_step_dir: str,
    ):
        mode = self.cfg.get("runtime", {}).get("mode", "mpc")

        if mode == "mpc":
            return self.mpc_planner.plan(world_state, memory, goal, episode_step_dir)

        if mode == "policy":
            if self.policy_planner is None:
                raise RuntimeError("Policy mode requested, but no policy planner is configured.")
            plan = self.policy_planner.plan(world_state, memory, goal, episode_step_dir=episode_step_dir)
            plan.metadata = {**(plan.metadata or {}), "runtime_mode": "policy"}
            return plan

        if mode in {"grounded_world_model", "world_model_policy"}:
            if self.grounded_world_model_planner is None:
                raise RuntimeError(
                    "Grounded world-model mode requested, but no grounded planner is configured."
                )
            plan = self.grounded_world_model_planner.plan(
                world_state,
                memory,
                goal,
                episode_step_dir=episode_step_dir,
            )
            plan.metadata = {
                **(plan.metadata or {}),
                "runtime_mode": "grounded_world_model",
            }
            return plan

        if mode == "hybrid":
            return self._plan_hybrid(world_state, memory, goal, episode_step_dir)

        raise RuntimeError(f"Unsupported runtime mode: {mode!r}")

    def _plan_hybrid(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        goal: GoalSpec,
        episode_step_dir: str,
    ):
        hybrid_cfg = self.cfg.get("hybrid", {})
        confidence_threshold = float(self.cfg.get("policy", {}).get("confidence_threshold", 0.75))

        if self.policy_planner is not None and hybrid_cfg.get("use_policy_when_confident", True):
            try:
                policy_plan = self.policy_planner.plan(
                    world_state,
                    memory,
                    goal,
                    episode_step_dir=episode_step_dir,
                )
                if (policy_plan.confidence or 0.0) >= confidence_threshold:
                    policy_plan.metadata = {
                        **(policy_plan.metadata or {}),
                        "runtime_mode": "hybrid",
                        "hybrid_decision": "policy",
                    }
                    return policy_plan
            except RuntimeError as exc:
                if not hybrid_cfg.get("fallback_to_mpc", True):
                    raise
                fallback_reason = str(exc)
            else:
                fallback_reason = "policy_confidence_below_threshold"
        else:
            fallback_reason = "policy_disabled"

        mpc_plan = self.mpc_planner.plan(world_state, memory, goal, episode_step_dir)
        mpc_plan.diagnostics["runtime_mode"] = "hybrid"
        mpc_plan.diagnostics["hybrid_decision"] = "mpc_fallback"
        mpc_plan.diagnostics["hybrid_fallback_reason"] = fallback_reason
        return mpc_plan
