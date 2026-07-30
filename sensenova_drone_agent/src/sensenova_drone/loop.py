from __future__ import annotations

from sensenova_drone.memory import MemoryEntry
from sensenova_drone.world_state import WorldState


class ClosedLoopAgent:
    def __init__(
        self,
        drone,
        observation_adapter,
        state_estimator,
        memory,
        world_model,
        planner,
        safety_shield,
        telemetry_logger,
        cfg,
    ):
        self.drone = drone
        self.observation_adapter = observation_adapter
        self.state_estimator = state_estimator
        self.memory = memory
        self.world_model = world_model
        self.planner = planner
        self.safety_shield = safety_shield
        self.telemetry_logger = telemetry_logger
        self.cfg = cfg

    async def step(self, goal):
        observation = await self.drone.read_observation()
        observation.frame_rgb = self.observation_adapter.preprocess_frame(
            observation.frame_rgb
        )

        if observation.pose is None:
            observation.pose = await self.state_estimator.estimate_pose()

        if observation.intrinsics is None:
            observation.intrinsics = await self.state_estimator.get_intrinsics()

        frame_path = self.telemetry_logger.save_real_frame(observation)

        encoding = self.world_model.encode_observation(
            observation.frame_rgb,
            frame_path=frame_path,
        )

        memory_entry = MemoryEntry(
            observation=observation,
            latent=encoding.latent,
            embedding=encoding.image_features,
            metadata={
                "frame_path": frame_path,
                "encoding_metadata": encoding.metadata,
                "source": "real_gazebo_camera",
            },
        )
        self.memory.append(memory_entry)

        world_state = WorldState(
            observation=observation,
            encoding=encoding,
            pose=observation.pose,
            intrinsics=observation.intrinsics,
            memory_size=len(self.memory),
        )

        episode_step_dir = self.telemetry_logger.make_step_dir()
        plan = self.planner.plan(
            world_state=world_state,
            memory=self.memory,
            goal=goal,
            episode_step_dir=episode_step_dir,
        )

        proposed_command = getattr(plan, "proposed_command", None)
        if proposed_command is None and hasattr(plan, "command"):
            proposed_command = plan.command

        safe_command = self.safety_shield.filter(
            proposed_command,
            observation=observation,
            memory=self.memory,
        )

        await self.drone.send_command(safe_command)

        self.telemetry_logger.log_step(
            observation=observation,
            world_state=world_state,
            plan=plan,
            executed_command=safe_command,
        )

        return plan, safe_command

    async def run(self, goal, max_steps: int):
        for _ in range(max_steps):
            await self.step(goal)
