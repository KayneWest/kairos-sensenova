from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
from PIL import Image

from sensenova_drone.action_conditioning import build_action_seed_video
from sensenova_drone.memory import RealObservationMemory
from sensenova_drone.scoring import GoalSpec
from sensenova_drone.world_state import (
    KairosActionCondition,
    ObservationEncoding,
    PredictedFuture,
    WorldState,
)


class KairosWorldModelAdapter(ABC):
    """
    Base adapter for Kairos/Sensenova as a world model.

    Required conceptual operations:
    1. encode real observation
    2. build or consume real memory
    3. rollout a temporary future under a candidate action condition
    4. optionally expose a decision-state h_t for a policy head
    """

    @abstractmethod
    def encode_observation(self, frame_rgb, frame_path: str | None = None) -> ObservationEncoding:
        raise NotImplementedError

    @abstractmethod
    def rollout_from_state(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        action_condition: KairosActionCondition,
        goal: GoalSpec,
        out_dir: str,
        return_type: str = "video",
    ) -> PredictedFuture:
        raise NotImplementedError

    def encode_observation_and_memory(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        goal: GoalSpec | None = None,
    ):
        _ = (world_state, memory, goal)
        return None


class SubprocessKairosAdapter(KairosWorldModelAdapter):
    """
    Fallback backend that drives Kairos through its public example entrypoint.

    This backend may not expose true latents. That is acceptable for the MVP:
    the planner still reasons in the correct conceptual loop by using the real
    frame path and a temporary rollout request per candidate action.
    """

    def __init__(self, cfg: dict, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def encode_observation(self, frame_rgb, frame_path: str | None = None) -> ObservationEncoding:
        _ = frame_rgb
        return ObservationEncoding(
            latent=None,
            image_features=None,
            frame_path=frame_path,
            metadata={"backend": "subprocess", "latent_available": False},
        )

    def rollout_from_state(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        action_condition: KairosActionCondition,
        goal: GoalSpec,
        out_dir: str,
        return_type: str = "video",
    ) -> PredictedFuture:
        _ = (memory, return_type)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        stdout_path = out_path / "stdout.log"
        stderr_path = out_path / "stderr.log"
        metadata_path = out_path / "metadata.json"

        try:
            input_frame_path = self._copy_input_frame(world_state, out_path)
        except Exception as exc:
            return self._write_failure(
                action_condition=action_condition,
                metadata_path=metadata_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                metadata={
                    "backend": "subprocess",
                    "error": f"Failed to prepare input frame: {exc}",
                },
            )

        request = self._load_request_template()
        request["prompt"] = self._build_prompt(goal.prompt, action_condition.prompt_suffix)
        request["input_image"] = str(input_frame_path)
        request["output_dir"] = str(out_path)
        conditioning_metadata = self._apply_action_conditioning(
            request=request,
            action_condition=action_condition,
            input_frame_path=input_frame_path,
            out_path=out_path,
        )

        request_path = out_path / "candidate_config.json"
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")

        base_metadata = {
            "action_sequence": [action.value for action in action_condition.action_sequence.actions],
            "prompt_suffix": action_condition.prompt_suffix,
            "camera_control_direction": action_condition.camera_control_direction,
            "camera_control_speed": action_condition.camera_control_speed,
            "camera_control_origin": action_condition.camera_control_origin,
            "input_frame": str(input_frame_path),
            "output_video": str(out_path / "output.mp4"),
            "candidate_config": str(request_path),
            "success": False,
            "return_type": return_type,
            "used_action_fields": self._used_action_fields(action_condition),
            "field_support": self._field_support_summary(),
            "action_condition_metadata": copy.deepcopy(action_condition.metadata),
            "conditioning_backend": conditioning_metadata["backend"],
            "conditioning_details": conditioning_metadata,
        }

        if not bool(self.cfg.get("kairos", {}).get("execute_subprocess", False)):
            base_metadata.update(
                {
                    "dry_run": True,
                    "note": "Candidate config created, but execute_subprocess=false.",
                }
            )
            metadata_path.write_text(json.dumps(base_metadata, indent=2), encoding="utf-8")
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            return PredictedFuture(
                action_condition=action_condition,
                success=False,
                metadata=base_metadata,
            )

        command = self._build_wrapper_command(request_path)
        if command is None:
            return self._write_failure(
                action_condition=action_condition,
                metadata_path=metadata_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                metadata={
                    **base_metadata,
                    "dry_run": True,
                    "error": "No wrapper command is configured.",
                },
            )

        cwd = Path(self.cfg.get("kairos", {}).get("repo_root", "/home/mkrzus/kairos-sensenova"))
        timeout_s = float(self.cfg.get("kairos", {}).get("subprocess_timeout_s", 600))
        try:
            result = subprocess.run(
                command,
                cwd=str(cwd),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            stdout_path.write_text(result.stdout, encoding="utf-8")
            stderr_path.write_text(result.stderr, encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text((exc.stdout or ""), encoding="utf-8")
            stderr_path.write_text((exc.stderr or ""), encoding="utf-8")
            return self._write_failure(
                action_condition=action_condition,
                metadata_path=metadata_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                metadata={
                    **base_metadata,
                    "command": command,
                    "timeout_s": timeout_s,
                    "error": str(exc),
                },
            )

        artifact_path = self._find_generated_artifact(out_path)
        success = result.returncode == 0 and artifact_path is not None
        metadata = {
            **base_metadata,
            "command": command,
            "returncode": result.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "success": success,
            "output_video": (
                str(artifact_path) if artifact_path and artifact_path.suffix.lower() == ".mp4" else None
            ),
            "final_frame_path": (
                str(artifact_path) if artifact_path and artifact_path.suffix.lower() != ".mp4" else None
            ),
        }
        if not success:
            metadata["error"] = self._summarize_process_failure(result.returncode, result.stdout, result.stderr)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return PredictedFuture(
            action_condition=action_condition,
            video_path=metadata["output_video"],
            final_frame_path=metadata.get("final_frame_path"),
            success=success,
            metadata=metadata,
        )

    def _load_request_template(self) -> dict[str, Any]:
        kairos_cfg = self.cfg.get("kairos", {})
        template_path = Path(
            kairos_cfg.get(
                "template_request_json",
                "/home/mkrzus/kairos-sensenova/examples/example_i2v.json",
            )
        )

        if template_path.is_file():
            try:
                return copy.deepcopy(json.loads(template_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse Kairos template request JSON at %s", template_path)

        return {
            "prompt": "",
            "input_image": "",
            "negative_prompt": "",
            "seed": 0,
            "tiled": True,
            "height": 704,
            "width": 1280,
            "num_frames": 81,
            "cfg_scale": 5,
            "use_prompt_rewriter": False,
            "output_dir": "output/i2v",
        }

    def _build_wrapper_command(self, request_path: Path) -> list[str] | None:
        kairos_cfg = self.cfg.get("kairos", {})
        wrapper_script = kairos_cfg.get(
            "wrapper_script",
            "/home/mkrzus/kairos-sensenova/sensenova_drone_agent/scripts/run_kairos_inference.sh",
        )
        config_file = kairos_cfg.get(
            "config_file",
            "/home/mkrzus/kairos-sensenova/kairos/configs/kairos_4b_config_DMD.py",
        )
        if not wrapper_script or not config_file:
            return None
        return [str(wrapper_script), str(request_path), str(config_file)]

    def _find_generated_artifact(self, out_dir: Path) -> Path | None:
        candidates = [out_dir / "output.mp4", out_dir / "output.jpg", out_dir / "output.png"]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        for pattern in ("*.mp4", "*.jpg", "*.png"):
            matches = sorted(out_dir.glob(pattern))
            if matches:
                for match in matches:
                    if match.name.startswith("input_frame"):
                        continue
                    return match
        return None

    def _copy_input_frame(self, world_state: WorldState, out_dir: Path) -> Path:
        frame_path = world_state.encoding.frame_path or world_state.observation.metadata.get("frame_path")
        if frame_path is None:
            raise RuntimeError("No real frame_path available for subprocess rollout.")

        source = Path(frame_path)
        if not source.exists():
            raise FileNotFoundError(f"Input frame does not exist: {source}")

        suffix = source.suffix or ".png"
        target = out_dir / f"input_frame{suffix}"
        shutil.copy2(source, target)
        return target

    def _build_prompt(self, goal_prompt: str, prompt_suffix: str) -> str:
        parts = [goal_prompt.strip(), prompt_suffix.strip()]
        return " ".join(part for part in parts if part)

    def _apply_action_conditioning(
        self,
        request: dict[str, Any],
        action_condition: KairosActionCondition,
        input_frame_path: Path,
        out_path: Path,
    ) -> dict[str, Any]:
        request.pop("metadata", None)
        request.pop("return_type", None)
        request.pop("input_video", None)

        runtime_camera_control_supported = self._camera_control_runtime_supported()
        fallback_enabled = bool(
            self.cfg.get("kairos", {}).get("enable_action_conditioned_input_video_fallback", False)
        )

        if fallback_enabled and not runtime_camera_control_supported:
            num_frames = int(request.get("num_frames", 49))
            seed_dir = out_path / "input_video_frames"
            seed_video = build_action_seed_video(
                image=Image.open(input_frame_path).convert("RGB"),
                actions=action_condition.action_sequence.actions,
                out_dir=seed_dir,
                num_frames=num_frames,
            )
            request["input_video"] = seed_video.frame_dir
            request.pop("camera_control_direction", None)
            request.pop("camera_control_speed", None)
            request.pop("camera_control_origin", None)
            return {
                "backend": "synthetic_input_video_fallback",
                "runtime_camera_control_supported": runtime_camera_control_supported,
                "fallback_reason": (
                    "Configured Kairos model does not expose a working control_adapter for camera_control_direction."
                ),
                "input_video": seed_video.frame_dir,
                "seed_video_metadata": seed_video.metadata,
            }

        if self._supports_field("camera_control_direction"):
            if action_condition.camera_control_direction is None:
                request.pop("camera_control_direction", None)
            else:
                request["camera_control_direction"] = action_condition.camera_control_direction

        if self._supports_field("camera_control_speed"):
            if action_condition.camera_control_direction is None or action_condition.camera_control_speed is None:
                request.pop("camera_control_speed", None)
            else:
                request["camera_control_speed"] = action_condition.camera_control_speed

        if self._supports_field("camera_control_origin"):
            if action_condition.camera_control_direction is None or action_condition.camera_control_origin is None:
                request.pop("camera_control_origin", None)
            else:
                request["camera_control_origin"] = list(action_condition.camera_control_origin)

        return {
            "backend": "native_camera_control",
            "runtime_camera_control_supported": runtime_camera_control_supported,
        }

    def _camera_control_runtime_supported(self) -> bool:
        kairos_cfg = self.cfg.get("kairos", {})
        if "runtime_camera_control_supported" in kairos_cfg:
            return bool(kairos_cfg["runtime_camera_control_supported"])

        config_file = kairos_cfg.get(
            "config_file",
            "/home/mkrzus/kairos-sensenova/kairos/configs/kairos_4b_config_DMD.py",
        )
        config_path = Path(config_file)
        try:
            config_text = config_path.read_text(encoding="utf-8")
        except OSError:
            return False

        has_image_input_true = bool(
            re.search(r"""["']has_image_input["']\s*:\s*True""", config_text)
        )
        add_control_adapter_true = bool(
            re.search(r"""["']add_control_adapter["']\s*:\s*True""", config_text)
        )
        if not has_image_input_true and not add_control_adapter_true:
            return False

        dit_path = Path(kairos_cfg.get("repo_root", "/home/mkrzus/kairos-sensenova")) / "kairos" / "modules" / "dits" / "kairos_dit.py"
        try:
            dit_text = dit_path.read_text(encoding="utf-8")
        except OSError:
            return False
        if "process_camera_coordinates" not in dit_text:
            return False
        return True

    def _field_support_summary(self) -> dict[str, bool]:
        return {
            "camera_control_direction": self._supports_field("camera_control_direction"),
            "camera_control_speed": self._supports_field("camera_control_speed"),
            "camera_control_origin": self._supports_field("camera_control_origin"),
            "input_video_json_entrypoint_supported": True,
            "runtime_camera_control_supported": self._camera_control_runtime_supported(),
        }

    def _supports_field(self, name: str) -> bool:
        supported = {
            "camera_control_direction",
            "camera_control_speed",
            "camera_control_origin",
            "input_image",
            "prompt",
            "negative_prompt",
            "num_frames",
            "height",
            "width",
            "seed",
        }
        return name in supported

    def _used_action_fields(self, action_condition: KairosActionCondition) -> dict[str, Any]:
        return {
            "camera_control_direction": action_condition.camera_control_direction,
            "camera_control_speed": action_condition.camera_control_speed if action_condition.camera_control_direction else None,
            "camera_control_origin": action_condition.camera_control_origin if action_condition.camera_control_direction else None,
        }

    def _summarize_process_failure(self, returncode: int, stdout: str, stderr: str) -> str:
        for stream in (stderr, stdout):
            for line in reversed(stream.splitlines()):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(("AttributeError:", "RuntimeError:", "TypeError:", "ValueError:", "AssertionError:")):
                    return f"Wrapper command failed (exit {returncode}): {stripped}"
        return f"Wrapper command returned {returncode} or no artifact was generated."

    def _write_failure(
        self,
        action_condition: KairosActionCondition,
        metadata_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        metadata: dict[str, Any],
    ) -> PredictedFuture:
        stdout_path.write_text(stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else "", encoding="utf-8")
        stderr_path.write_text(stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else "", encoding="utf-8")
        payload = {"success": False, **metadata}
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return PredictedFuture(
            action_condition=action_condition,
            success=False,
            metadata=payload,
        )


class PythonKairosAdapter(KairosWorldModelAdapter):
    """
    Future backend for native Python-side Kairos access.

    This is scaffolded now so later latent/state access can be dropped in
    without changing the planner or loop interfaces.
    """

    def __init__(self, cfg: dict, logger: logging.Logger | None = None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def encode_observation(self, frame_rgb, frame_path: str | None = None) -> ObservationEncoding:
        _ = (frame_rgb, frame_path)
        return ObservationEncoding(
            latent=None,
            image_features=None,
            frame_path=frame_path,
            metadata={
                "backend": "python_api",
                "latent_available": False,
                "status": "TODO: use Kairos/Wan VAE to encode frame into latent.",
            },
        )

    def encode_observation_and_memory(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        goal: GoalSpec | None = None,
    ):
        _ = (world_state, memory, goal)
        return None

    def rollout_from_state(
        self,
        world_state: WorldState,
        memory: RealObservationMemory,
        action_condition: KairosActionCondition,
        goal: GoalSpec,
        out_dir: str,
        return_type: str = "video",
    ) -> PredictedFuture:
        _ = (world_state, memory, action_condition, goal, out_dir, return_type)
        raise NotImplementedError(
            "TODO: call Kairos pipeline directly instead of subprocess for native latent/state rollouts."
        )
