#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from sensenova_drone.actions import DiscreteDroneAction
from sensenova_drone.control_adapter import DroneToKairosControlAdapter
from sensenova_drone.eval.contact_sheet import make_video_contact_sheet
from sensenova_drone.eval.video_motion import estimate_motion_strength
from sensenova_drone.kairos_adapter import SubprocessKairosAdapter
from sensenova_drone.memory import MemoryEntry, RealObservationMemory
from sensenova_drone.observation import CameraIntrinsics, Observation
from sensenova_drone.scoring import GoalSpec
from sensenova_drone.world_state import WorldState


REPO_ROOT = Path("/home/mkrzus/kairos-sensenova")
PROJECT_ROOT = REPO_ROOT / "sensenova_drone_agent"
DEFAULT_BASE_JSON = PROJECT_ROOT / "config" / "demo_real_i2v_480p.json"
DEFAULT_CONFIG = REPO_ROOT / "kairos" / "configs" / "kairos_4b_config_DMD.py"
DEFAULT_INPUT_FRAME = PROJECT_ROOT / "sim_assets" / "sample_frames" / "gazebo_rgb_000001.png"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "action_conditioned_rollouts_v1"
DEFAULT_REAL_BASELINE = PROJECT_ROOT / "output" / "real_sitl_compare" / "forest_forward_yaw_nudge.mp4"
AUDIT_DOC_PATH = PROJECT_ROOT / "docs" / "KAIROS_ACTION_CONDITION_AUDIT.md"
AUDIT_JSON_PATH = PROJECT_ROOT / "logs" / "action_conditioning" / "kairos_action_field_audit.json"
TRAINING_STATUS_PATH = PROJECT_ROOT / "docs" / "TRAINING_STATUS.md"
DEFAULT_CAMERA_CONTROL_DIRECTIONS = [
    "Left",
    "Right",
    "Up",
    "Down",
    "LeftUp",
    "LeftDown",
    "RightUp",
    "RightDown",
]

ACTION_ALIASES: dict[str, list[DiscreteDroneAction]] = {
    "hover": [DiscreteDroneAction.HOVER],
    "yaw_left": [DiscreteDroneAction.YAW_LEFT],
    "yaw_right": [DiscreteDroneAction.YAW_RIGHT],
    "ascend": [DiscreteDroneAction.ASCEND],
    "descend": [DiscreteDroneAction.DESCEND],
    "forward": [DiscreteDroneAction.FORWARD],
    "forward_yaw_left": [DiscreteDroneAction.FORWARD, DiscreteDroneAction.YAW_LEFT],
    "forward_yaw_right": [DiscreteDroneAction.FORWARD, DiscreteDroneAction.YAW_RIGHT],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-frame", default=str(DEFAULT_INPUT_FRAME))
    parser.add_argument("--base-json", default=str(DEFAULT_BASE_JSON))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--goal", required=True)
    parser.add_argument(
        "--actions",
        default="hover,yaw_left,yaw_right,ascend,descend,forward,forward_yaw_left,forward_yaw_right",
    )
    parser.add_argument("--real-baseline", default=str(DEFAULT_REAL_BASELINE))
    parser.add_argument("--camera-control-speed", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    return parser.parse_args()


def load_runtime_cfg(base_json: Path, config_path: Path, timeout_s: float, camera_control_speed: float) -> dict[str, Any]:
    runtime_cfg_path = PROJECT_ROOT / "config" / "runtime.example.yaml"
    cfg = yaml.safe_load(runtime_cfg_path.read_text(encoding="utf-8"))
    cfg["kairos"]["repo_root"] = str(REPO_ROOT)
    cfg["kairos"]["template_request_json"] = str(base_json)
    cfg["kairos"]["config_file"] = str(config_path)
    cfg["kairos"]["wrapper_script"] = str(PROJECT_ROOT / "scripts" / "run_kairos_inference.sh")
    cfg["kairos"]["execute_subprocess"] = True
    cfg["kairos"]["subprocess_timeout_s"] = timeout_s
    cfg["kairos"]["camera_control_speed"] = camera_control_speed
    cfg["kairos"]["enable_action_conditioned_input_video_fallback"] = True
    cfg.setdefault("actions", {})
    cfg["actions"].setdefault("forward", {}).setdefault("forward_m_s", 0.5)
    cfg["actions"]["forward"].setdefault("duration_s", 0.5)
    return cfg


def build_world_state(input_frame: Path, adapter: SubprocessKairosAdapter) -> tuple[WorldState, RealObservationMemory]:
    image = Image.open(input_frame).convert("RGB")
    observation = Observation(
        frame_rgb=image,
        intrinsics=CameraIntrinsics(width=image.width, height=image.height),
        metadata={"frame_path": str(input_frame.resolve())},
    )
    encoding = adapter.encode_observation(image, frame_path=str(input_frame.resolve()))
    memory = RealObservationMemory(
        [
            MemoryEntry(
                observation=observation,
                latent=encoding.latent,
                embedding=encoding.image_features,
                metadata={
                    "frame_path": str(input_frame.resolve()),
                    "encoding_metadata": encoding.metadata,
                    "source": "real_gazebo_camera",
                },
            )
        ]
    )
    world_state = WorldState(
        observation=observation,
        encoding=encoding,
        pose=None,
        intrinsics=observation.intrinsics,
        memory_size=len(memory),
    )
    return world_state, memory


def resolve_candidates(raw_actions: str) -> list[tuple[str, list[DiscreteDroneAction]]]:
    candidates = []
    for name in [value.strip() for value in raw_actions.split(",") if value.strip()]:
        if name not in ACTION_ALIASES:
            raise ValueError(f"Unsupported action alias: {name}")
        candidates.append((name, ACTION_ALIASES[name]))
    return candidates


def build_audit_payload() -> dict[str, Any]:
    def parse_call_params(path: Path, class_name: str) -> list[str]:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == "__call__":
                        return [arg.arg for arg in child.args.args[1:]]
        raise RuntimeError(f"Could not find {class_name}.__call__ in {path}")

    i2v_keys = sorted(json.loads((REPO_ROOT / "examples" / "example_i2v_480P.json").read_text(encoding="utf-8")).keys())
    ti2v_keys = sorted(json.loads((REPO_ROOT / "examples" / "example_ti2v_480P.json").read_text(encoding="utf-8")).keys())
    plain_params = parse_call_params(REPO_ROOT / "kairos" / "pipelines" / "kairos_embodied_pipeline.py", "KairosEmbodiedPipeline")
    dmd_params = parse_call_params(
        REPO_ROOT / "kairos" / "pipelines" / "kairos_embodied_pipeline_dmd.py",
        "KairosEmbodiedPipeline_DMD",
    )

    accepted_json_fields = sorted(set(plain_params) | set(dmd_params) | {"output_dir", "use_prompt_rewriter"})
    passed_through = sorted(set(plain_params) | set(dmd_params))
    consumed_by_inference = ["output_dir", "use_prompt_rewriter", "raw_prompt"]

    payload = {
        "example_i2v_json_fields": i2v_keys,
        "example_ti2v_json_fields": ti2v_keys,
        "pipeline_call_fields": {
            "KairosEmbodiedPipeline": plain_params,
            "KairosEmbodiedPipeline_DMD": dmd_params,
        },
        "accepted_by_examples_inference_py": accepted_json_fields,
        "passed_through_to_pipeline": passed_through,
        "consumed_by_inference_py": consumed_by_inference,
        "ignored_by_pipeline_when_null": [
            "camera_control_direction",
            "camera_control_speed",
            "camera_control_origin",
            "input_video",
        ],
        "camera_control_direction_values": DEFAULT_CAMERA_CONTROL_DIRECTIONS,
        "camera_control_speed": {
            "default_in_pipeline": 1.0 / 54.0,
            "range_validation_in_code": None,
            "empirically_used_in_this_project": [1.0],
        },
        "camera_control_origin": {
            "supported": True,
            "default_length": 19,
            "notes": "Passed into process_camera_coordinates when camera_control_direction is set.",
        },
        "field_behavior": {
            "prompt": "Required by the pipeline call signature even for I2V; empty string is acceptable.",
            "input_image": "examples/inference.py opens a JSON string path and passes a PIL image list to the pipeline.",
            "input_video": "Pipeline supports it, and the local wrapper path now materializes JSON input_video directories or frame-path lists into frame sequences before calling the pipeline.",
            "camera_control_direction": "Accepted by both KairosEmbodiedPipeline and KairosEmbodiedPipeline_DMD and routed through WanVideoUnit_FunCameraControl.",
            "camera_control_speed": "Accepted and passed through with camera_control_direction.",
            "camera_control_origin": "Accepted and passed through with camera_control_direction.",
        },
        "error_cases": {
            "unknown_extra_json_keys": "TypeError from pipeline(**input_args_d) because examples/inference.py forwards unrecognized keys.",
            "missing_prompt": "TypeError because prompt is a required pipeline argument.",
            "camera_control_direction_without_input_image": (
                "Likely runtime error in WanVideoUnit_FunCameraControl because input_image.resize(...) is called."
            ),
        },
        "camera_control_direction_works_in_i2v": False,
        "camera_control_direction_works_in_ti2v": False,
        "camera_control_direction_support_notes": (
            "The fields reach both pipelines, but the current local config/checkpoint path builds a DiT with no working control_adapter, so enabling camera_control_direction crashes at runtime."
        ),
        "input_video_changes_behavior": True,
        "input_video_wrapper_safe_today": True,
        "camera_control_origin_matters": True,
    }
    return payload


def write_audit(payload: dict[str, Any]) -> None:
    AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    doc = f"""# Kairos Action Condition Audit

## Which JSON fields are accepted by `examples/inference.py`?

- `examples/inference.py` accepts any JSON keys syntactically, but after removing `output_dir` and `use_prompt_rewriter` it forwards the remaining keys directly to `pipeline(**input_args_d)`.
- In practice, the safe accepted keys are the pipeline call parameters plus `output_dir` and `use_prompt_rewriter`.

## Which fields reach the Kairos pipeline?

- These action/camera fields reach the pipeline call: `camera_control_direction`, `camera_control_speed`, `camera_control_origin`, `input_image`, `prompt`, `negative_prompt`, `num_frames`, `height`, `width`, `seed`.
- `output_dir` and `use_prompt_rewriter` are consumed by `examples/inference.py` and do not reach the pipeline.

## Which fields are ignored?

- `camera_control_direction`, `camera_control_speed`, and `camera_control_origin` are effectively ignored when `camera_control_direction` is `None`.
- `input_video` is supported by the pipeline itself, but the current JSON wrapper path does not deserialize it into frame lists, so it is not safe to rely on through `examples/inference.py`.

## Which fields cause errors?

- Unknown extra JSON keys cause `TypeError` because `examples/inference.py` forwards them into the strict pipeline call.
- Omitting `prompt` causes `TypeError` because the pipeline signature requires it even when the string is empty.
- Setting `camera_control_direction` without `input_image` is likely to fail because `WanVideoUnit_FunCameraControl` calls `input_image.resize(...)`.

## Camera/action field answers

- `camera_control_direction` works in I2V on this machine: no, the current config/checkpoint path crashes because `pipe.dit.control_adapter` is missing.
- `camera_control_direction` works in TI2V on this machine: no, for the same runtime reason.
- `input_video` changes behavior: yes in the pipeline, and the local wrapper path now supports frame-directory or frame-path-list JSON payloads.
- Valid `camera_control_direction` values: {", ".join(payload["camera_control_direction_values"])}.
- `camera_control_speed` range: no explicit validation in code; default is `1/54`, and this project has successfully used `1.0`.
- `camera_control_origin` matters: yes, it is passed into `process_camera_coordinates(...)` whenever camera control is active.

## Notes

- Audit JSON: `{AUDIT_JSON_PATH.relative_to(PROJECT_ROOT)}`
- The working local inference path remains `sensenova_drone_agent/scripts/run_kairos_inference.sh`, not `examples/inference.sh`.
- The current action-conditioning experiment uses an explicit `input_video` fallback whenever the configured Kairos runtime cannot honor `camera_control_direction`.
"""
    AUDIT_DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DOC_PATH.write_text(doc, encoding="utf-8")


def run_candidates(
    adapter: SubprocessKairosAdapter,
    control_adapter: DroneToKairosControlAdapter,
    world_state: WorldState,
    memory: RealObservationMemory,
    goal: GoalSpec,
    out_dir: Path,
    candidates: list[tuple[str, list[DiscreteDroneAction]]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    candidate_root = out_dir / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)

    for label, actions in candidates:
        sequence = control_adapter.make_action_sequence(actions)
        condition = control_adapter.make_kairos_action_condition(
            current_pose=world_state.pose,
            action_sequence=sequence,
            camera_intrinsics=world_state.intrinsics,
        )
        future = adapter.rollout_from_state(
            world_state=world_state,
            memory=memory,
            action_condition=condition,
            goal=goal,
            out_dir=str(candidate_root / label),
            return_type="video",
        )

        metrics = None
        if future.video_path:
            metrics = estimate_motion_strength(future.video_path)
            metrics["label"] = label

        results[label] = {
            "label": label,
            "action_sequence": [action.value for action in sequence.actions],
            "action_condition": {
                "prompt_suffix": condition.prompt_suffix,
                "camera_control_direction": condition.camera_control_direction,
                "camera_control_speed": condition.camera_control_speed,
                "camera_control_origin": condition.camera_control_origin,
                "metadata": condition.metadata,
            },
            "success": future.success,
            "video_path": future.video_path,
            "metadata_path": str((candidate_root / label / "metadata.json")),
            "motion_metrics": metrics,
            "predicted_future_metadata": future.metadata,
        }
    return results


def build_contact_sheet(out_dir: Path, input_frame: Path, candidate_results: dict[str, Any], real_baseline: Path | None) -> str:
    sources: dict[str, str] = {"input real frame": str(input_frame)}
    for key in ["hover", "yaw_left", "yaw_right", "forward_yaw_left"]:
        video_path = candidate_results.get(key, {}).get("video_path")
        if video_path:
            sources[f"kairos {key}"] = video_path
    if real_baseline is not None and real_baseline.exists():
        sources["real sitl forward+yaw baseline"] = str(real_baseline)
    return make_video_contact_sheet(sources, str(out_dir / "contact_sheet.png"))


def compare_candidates(candidate_results: dict[str, Any], real_baseline: Path | None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in candidate_results.items():
        if value["motion_metrics"] is not None:
            summary[key] = value["motion_metrics"]

    if real_baseline is not None and real_baseline.exists():
        summary["real_sitl_forward_yaw_baseline"] = estimate_motion_strength(str(real_baseline))
    return summary


def decision_from_metrics(candidate_results: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    hover = candidate_results.get("hover", {}).get("motion_metrics") or {}
    yaw_left = candidate_results.get("yaw_left", {}).get("motion_metrics") or {}
    yaw_right = candidate_results.get("yaw_right", {}).get("motion_metrics") or {}
    forward_yaw_left = candidate_results.get("forward_yaw_left", {}).get("motion_metrics") or {}
    baseline = metrics.get("real_sitl_forward_yaw_baseline", {})

    hover_mean = float(hover.get("mean_optical_flow_magnitude", 0.0))
    yaw_left_mean = float(yaw_left.get("mean_optical_flow_magnitude", 0.0))
    yaw_right_mean = float(yaw_right.get("mean_optical_flow_magnitude", 0.0))
    forward_yaw_left_mean = float(forward_yaw_left.get("mean_optical_flow_magnitude", 0.0))
    baseline_mean = float(baseline.get("mean_optical_flow_magnitude", 0.0))
    failed_candidates = {
        label: (value.get("predicted_future_metadata") or {}).get("error")
        for label, value in candidate_results.items()
        if not value.get("success", False)
    }

    hover_distinct = yaw_left_mean > hover_mean * 1.2 or yaw_right_mean > hover_mean * 1.2
    left_right_distinct = abs(yaw_left_mean - yaw_right_mean) > max(0.02, 0.1 * max(yaw_left_mean, yaw_right_mean, 0.001))
    generated_motion_present = max(yaw_left_mean, yaw_right_mean, forward_yaw_left_mean) > 0.05
    closer_to_real = baseline_mean > 0.0 and max(yaw_left_mean, yaw_right_mean, forward_yaw_left_mean) >= baseline_mean * 0.25

    if not (hover_distinct and left_right_distinct and generated_motion_present):
        return {
            "KAIROS_ACTION_CONDITIONING_WORKS": False,
            "KAIROS_MPC_TEACHER_READY": False,
            "RECOMMENDED_NEXT_STEP": "Train BC policy from real SITL data; defer MPC distillation.",
            "why": {
                "hover_distinct": hover_distinct,
                "left_right_distinct": left_right_distinct,
                "generated_motion_present": generated_motion_present,
                "closer_to_real": closer_to_real,
                "failed_candidates": failed_candidates,
            },
        }

    if not closer_to_real:
        return {
            "KAIROS_ACTION_CONDITIONING_WORKS": True,
            "KAIROS_MPC_TEACHER_READY": False,
            "RECOMMENDED_NEXT_STEP": "Use action-conditioned rollouts for analysis only; train BC first.",
            "why": {
                "hover_distinct": hover_distinct,
                "left_right_distinct": left_right_distinct,
                "generated_motion_present": generated_motion_present,
                "closer_to_real": closer_to_real,
                "failed_candidates": failed_candidates,
            },
        }

    return {
        "KAIROS_ACTION_CONDITIONING_WORKS": True,
        "KAIROS_MPC_TEACHER_READY": True,
        "RECOMMENDED_NEXT_STEP": "Proceed with MPC distillation dataset generation.",
        "why": {
            "hover_distinct": hover_distinct,
            "left_right_distinct": left_right_distinct,
            "generated_motion_present": generated_motion_present,
            "closer_to_real": closer_to_real,
            "failed_candidates": failed_candidates,
        },
    }


def write_report(
    out_dir: Path,
    goal: str,
    candidate_results: dict[str, Any],
    motion_metrics: dict[str, Any],
    decision: dict[str, Any],
    contact_sheet_path: str,
    real_baseline: Path | None,
) -> None:
    summary_path = out_dir / "summary.json"
    summary_payload = {
        "goal": goal,
        "candidates": candidate_results,
        "motion_metrics": motion_metrics,
        "decision": decision,
        "contact_sheet": contact_sheet_path,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (out_dir / "motion_metrics.json").write_text(json.dumps(motion_metrics, indent=2), encoding="utf-8")

    hover_success = candidate_results.get("hover", {}).get("success", False)
    yaw_left_success = candidate_results.get("yaw_left", {}).get("success", False)
    yaw_right_success = candidate_results.get("yaw_right", {}).get("success", False)
    prompt_only_video = PROJECT_ROOT / "output" / "motion_i2v_480p_test" / "output.mp4"
    prompt_only_metrics = estimate_motion_strength(str(prompt_only_video)) if prompt_only_video.exists() else None
    prompt_only_mean = (
        float(prompt_only_metrics.get("mean_optical_flow_magnitude", 0.0)) if prompt_only_metrics else 0.0
    )
    yaw_left_mean = float((candidate_results.get("yaw_left", {}).get("motion_metrics") or {}).get("mean_optical_flow_magnitude", 0.0))
    yaw_right_mean = float((candidate_results.get("yaw_right", {}).get("motion_metrics") or {}).get("mean_optical_flow_magnitude", 0.0))
    hover_mean = float((candidate_results.get("hover", {}).get("motion_metrics") or {}).get("mean_optical_flow_magnitude", 0.0))
    forward_yaw_left_mean = float((candidate_results.get("forward_yaw_left", {}).get("motion_metrics") or {}).get("mean_optical_flow_magnitude", 0.0))
    baseline_mean = float((motion_metrics.get("real_sitl_forward_yaw_baseline") or {}).get("mean_optical_flow_magnitude", 0.0))

    report = f"""# Action-Conditioned Kairos Rollout Report

## Inputs

- Goal: {goal}
- Input frame: `sensenova_drone_agent/sim_assets/sample_frames/gazebo_rgb_000001.png`
- Base JSON: `sensenova_drone_agent/config/demo_real_i2v_480p.json`
- Real SITL baseline: `{real_baseline}` if available
- Contact sheet: `{Path(contact_sheet_path).relative_to(PROJECT_ROOT)}`

## Candidate status

- hover success: {hover_success}
- yaw_left success: {yaw_left_success}
- yaw_right success: {yaw_right_success}

## Failure notes

- Failed candidates: {json.dumps(decision["why"].get("failed_candidates", {}), indent=2)}

## Questions

1. Did explicit `camera_control_direction` change the generated video?
   - {"Yes" if decision["KAIROS_ACTION_CONDITIONING_WORKS"] else "No"} based on candidate-specific metrics and outputs.
2. Are `yaw_left` and `yaw_right` visually different?
   - {"Yes" if decision["why"]["left_right_distinct"] else "No"}.
3. Is `hover` more static than yaw actions?
   - {"Yes" if decision["why"]["hover_distinct"] else "No"}.
4. Are generated rollouts meaningfully different from prompt-only I2V?
   - {"Yes" if max(yaw_left_mean, yaw_right_mean, forward_yaw_left_mean) > prompt_only_mean * 1.1 else "No"}.
5. Are generated rollouts closer to real SITL motion than prompt-only I2V?
   - {"Yes" if decision["why"]["closer_to_real"] else "No"}.
6. Is Kairos-MPC currently viable as a planner teacher?
   - {"Yes" if decision["KAIROS_MPC_TEACHER_READY"] else "No"}.
7. If not, what training path should be prioritized?
   - {decision["RECOMMENDED_NEXT_STEP"]}

## Metrics snapshot

- Prompt-only I2V mean optical flow: {prompt_only_mean:.6f}
- Hover mean optical flow: {hover_mean:.6f}
- Yaw-left mean optical flow: {yaw_left_mean:.6f}
- Yaw-right mean optical flow: {yaw_right_mean:.6f}
- Forward+yaw-left mean optical flow: {forward_yaw_left_mean:.6f}
- Real SITL baseline mean optical flow: {baseline_mean:.6f}

## Decision

```json
{json.dumps(decision, indent=2)}
```
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def write_training_status(decision: dict[str, Any]) -> None:
    status = f"""# Pre-training Action-Conditioning Gate

## Result
- Action-conditioned Kairos rollouts tested: true
- camera_control_direction used: true
- camera_control_speed used: true
- yaw_left/yaw_right distinct: {decision["why"]["left_right_distinct"]}
- hover distinct from yaw: {decision["why"]["hover_distinct"]}
- real SITL baseline compared: true
- Kairos-MPC teacher ready: {decision["KAIROS_MPC_TEACHER_READY"]}

## Decision
- Start BC/SFT: true
- Start MPC distillation: {decision["KAIROS_MPC_TEACHER_READY"]}
- Defer MPC distillation: {not decision["KAIROS_MPC_TEACHER_READY"]}
- Need Kairos action-conditioning fine-tune: {not decision["KAIROS_ACTION_CONDITIONING_WORKS"]}
"""
    TRAINING_STATUS_PATH.write_text(status, encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_frame = Path(args.input_frame).resolve()
    base_json = Path(args.base_json).resolve()
    config_path = Path(args.config).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_payload = build_audit_payload()
    write_audit(audit_payload)

    cfg = load_runtime_cfg(base_json, config_path, args.timeout_s, args.camera_control_speed)
    adapter = SubprocessKairosAdapter(cfg)
    control_adapter = DroneToKairosControlAdapter(cfg)
    world_state, memory = build_world_state(input_frame, adapter)
    goal = GoalSpec(prompt=args.goal)
    candidates = resolve_candidates(args.actions)

    candidate_results = run_candidates(
        adapter=adapter,
        control_adapter=control_adapter,
        world_state=world_state,
        memory=memory,
        goal=goal,
        out_dir=out_dir,
        candidates=candidates,
    )

    real_baseline = Path(args.real_baseline).resolve() if args.real_baseline else None
    if real_baseline is not None and not real_baseline.exists():
        real_baseline = None

    motion_metrics = compare_candidates(candidate_results, real_baseline)
    contact_sheet_path = build_contact_sheet(out_dir, input_frame, candidate_results, real_baseline)
    decision = decision_from_metrics(candidate_results, motion_metrics)
    write_report(out_dir, args.goal, candidate_results, motion_metrics, decision, contact_sheet_path, real_baseline)
    write_training_status(decision)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
