from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from sensenova_drone.actions import DiscreteDroneAction, DroneCommand, coerce_discrete_action
from sensenova_drone.observation import CameraIntrinsics, Pose


ACTION_VOCAB = [action.value for action in DiscreteDroneAction]
ACTION_TO_INDEX = {action: index for index, action in enumerate(ACTION_VOCAB)}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BCEpisodeStep:
    episode_id: str
    step_index: int
    action: DiscreteDroneAction
    command: DroneCommand
    image_path: str
    next_image_path: str | None = None
    timestamp_s: float | None = None
    pose: Pose | None = None
    intrinsics: CameraIntrinsics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "action": self.action.value,
            "action_index": ACTION_TO_INDEX[self.action.value],
            "command": _command_to_dict(self.command),
            "image_path": self.image_path,
            "next_image_path": self.next_image_path,
            "timestamp_s": self.timestamp_s,
            "pose": _pose_to_dict(self.pose),
            "intrinsics": _intrinsics_to_dict(self.intrinsics),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BCEpisodeStep":
        return cls(
            episode_id=str(value["episode_id"]),
            step_index=int(value["step_index"]),
            action=coerce_discrete_action(value["action"]),
            command=_command_from_dict(value["command"]),
            image_path=str(value["image_path"]),
            next_image_path=value.get("next_image_path"),
            timestamp_s=_maybe_float(value.get("timestamp_s")),
            pose=Pose.from_mapping(value["pose"]) if value.get("pose") else None,
            intrinsics=CameraIntrinsics.from_mapping(value["intrinsics"]) if value.get("intrinsics") else None,
            metadata=dict(value.get("metadata", {})),
        )


def save_episode_step(step_dir: str | Path, step: BCEpisodeStep) -> str:
    step_path = Path(step_dir)
    step_path.mkdir(parents=True, exist_ok=True)
    output_path = step_path / "step.json"
    output_path.write_text(json.dumps(step.to_dict(), indent=2), encoding="utf-8")
    return str(output_path)


def load_episode_steps(episode_dir: str | Path) -> list[BCEpisodeStep]:
    episode_path = Path(episode_dir)
    step_paths = sorted(episode_path.glob("step_*/step.json"))
    if not step_paths:
        step_paths = sorted(episode_path.rglob("step.json"))
    steps: list[BCEpisodeStep] = []
    for step_path in step_paths:
        step = BCEpisodeStep.from_dict(json.loads(step_path.read_text(encoding="utf-8")))
        step.image_path = _resolve_record_path(step.image_path, episode_path)
        step.next_image_path = _resolve_record_path(step.next_image_path, episode_path)
        steps.append(step)
    return steps


def export_bc_manifest(
    episodes_root: str | Path,
    out_jsonl: str | Path,
    *,
    val_ratio: float = 0.1,
    summary_json: str | Path | None = None,
    include_worlds: set[str] | None = None,
    required_decision_family: str | None = None,
    allowed_decision_families: set[str] | None = None,
    require_decision_rich: bool = False,
    allowed_actions: set[str] | None = None,
    followthrough_after_families: set[str] | None = None,
    followthrough_family: str | None = None,
    followthrough_actions: set[str] | None = None,
    followthrough_steps: int = 0,
) -> dict[str, Any]:
    episodes_path = Path(episodes_root)
    if not episodes_path.exists():
        raise FileNotFoundError(f"Episodes root does not exist: {episodes_path}")
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    counts_by_action = {action.value: 0 for action in DiscreteDroneAction}
    episodes_with_steps: list[tuple[Path, list[BCEpisodeStep]]] = []
    selected_episode_steps: list[tuple[str, list[tuple[BCEpisodeStep, str]]]] = []
    counts_by_teacher_reason: dict[str, int] = {}
    counts_by_decision_family: dict[str, int] = {}
    counts_by_target_family: dict[str, int] = {}
    counts_by_world: dict[str, int] = {}
    counts_by_scenario: dict[str, int] = {}
    counts_by_selection_mode: dict[str, int] = {}
    decision_rich_examples = 0
    branch_scores: list[float] = []

    for episode_dir in sorted(path for path in episodes_path.iterdir() if path.is_dir()):
        steps = load_episode_steps(episode_dir)
        if not steps:
            continue
        episodes_with_steps.append((episode_dir, steps))

    for _, steps in episodes_with_steps:
        selected_steps = _select_episode_steps(
            steps,
            include_worlds=include_worlds,
            required_decision_family=required_decision_family,
            allowed_decision_families=allowed_decision_families,
            require_decision_rich=require_decision_rich,
            allowed_actions=allowed_actions,
            followthrough_after_families=followthrough_after_families,
            followthrough_family=followthrough_family,
            followthrough_actions=followthrough_actions,
            followthrough_steps=followthrough_steps,
        )
        if selected_steps:
            selected_episode_steps.append((steps[0].episode_id, selected_steps))

    split_map = episode_split_map(
        [episode_id for episode_id, _ in selected_episode_steps],
        val_ratio=val_ratio,
    )

    for episode_id, selected_steps in selected_episode_steps:
        split = split_map[episode_id]
        for step, selection_mode in selected_steps:
            record = step.to_dict()
            record["split"] = split
            record["selection_mode"] = selection_mode
            records.append(record)
            counts_by_action[step.action.value] += 1
            counts_by_selection_mode[selection_mode] = counts_by_selection_mode.get(selection_mode, 0) + 1
            teacher = dict(step.metadata.get("teacher", {}))
            world_label = step.metadata.get("world_label")
            scenario_label = step.metadata.get("scenario_label")
            if world_label:
                counts_by_world[str(world_label)] = counts_by_world.get(str(world_label), 0) + 1
            if scenario_label:
                counts_by_scenario[str(scenario_label)] = counts_by_scenario.get(str(scenario_label), 0) + 1
            reason = teacher.get("reason")
            if reason:
                counts_by_teacher_reason[str(reason)] = counts_by_teacher_reason.get(str(reason), 0) + 1
            decision_profile = dict(teacher.get("decision_profile", {}))
            family = decision_profile.get("family")
            if family:
                counts_by_decision_family[str(family)] = counts_by_decision_family.get(str(family), 0) + 1
            target_family = decision_profile.get("target_family")
            if target_family:
                counts_by_target_family[str(target_family)] = counts_by_target_family.get(str(target_family), 0) + 1
            if bool(decision_profile.get("decision_rich", False)):
                decision_rich_examples += 1
            branch_score = decision_profile.get("branch_score")
            if branch_score is not None:
                branch_scores.append(float(branch_score))

    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary = {
        "episodes_root": str(episodes_path.resolve()),
        "manifest_path": str(out_path.resolve()),
        "num_source_episodes": len(episodes_with_steps),
        "num_episodes": len(selected_episode_steps),
        "num_examples": len(records),
        "counts_by_action": counts_by_action,
        "train_examples": sum(1 for record in records if record["split"] == "train"),
        "val_examples": sum(1 for record in records if record["split"] == "val"),
        "val_episode_count": sum(1 for split in split_map.values() if split == "val"),
        "action_vocab": ACTION_VOCAB,
        "filters": {
            "include_worlds": sorted(include_worlds) if include_worlds else [],
            "required_decision_family": required_decision_family,
            "allowed_decision_families": sorted(allowed_decision_families) if allowed_decision_families else [],
            "require_decision_rich": bool(require_decision_rich),
            "allowed_actions": sorted(allowed_actions) if allowed_actions else [],
            "followthrough_after_families": sorted(followthrough_after_families) if followthrough_after_families else [],
            "followthrough_family": followthrough_family,
            "followthrough_actions": sorted(followthrough_actions) if followthrough_actions else [],
            "followthrough_steps": int(max(0, followthrough_steps)),
        },
        "counts_by_selection_mode": counts_by_selection_mode,
        "counts_by_teacher_reason": counts_by_teacher_reason,
        "counts_by_decision_family": counts_by_decision_family,
        "counts_by_target_family": counts_by_target_family,
        "counts_by_world": counts_by_world,
        "counts_by_scenario": counts_by_scenario,
        "decision_rich_examples": decision_rich_examples,
        "decision_rich_fraction": (
            float(decision_rich_examples) / float(len(records))
            if records
            else 0.0
        ),
        "mean_branch_score": (
            sum(branch_scores) / len(branch_scores)
            if branch_scores
            else None
        ),
    }
    if summary_json is not None:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def episode_split(episode_id: str, *, val_ratio: float = 0.1) -> str:
    return episode_split_map([episode_id], val_ratio=val_ratio)[episode_id]


def episode_split_map(episode_ids: list[str], *, val_ratio: float = 0.1) -> dict[str, str]:
    unique_ids = list(dict.fromkeys(episode_ids))
    if not unique_ids:
        return {}

    if val_ratio <= 0.0 or len(unique_ids) == 1:
        return {episode_id: "train" for episode_id in unique_ids}

    scored_ids = sorted(
        ((_episode_hash_fraction(episode_id), episode_id) for episode_id in unique_ids),
        key=lambda item: (item[0], item[1]),
    )
    requested_val = int(math.ceil(len(unique_ids) * float(val_ratio)))
    num_val = min(len(unique_ids) - 1, max(1, requested_val))
    val_ids = {episode_id for _, episode_id in scored_ids[:num_val]}
    return {
        episode_id: ("val" if episode_id in val_ids else "train")
        for episode_id in unique_ids
    }


def _command_to_dict(command: DroneCommand) -> dict[str, Any]:
    payload = {
        "forward_m_s": command.forward_m_s,
        "right_m_s": command.right_m_s,
        "down_m_s": command.down_m_s,
        "yawspeed_deg_s": command.yawspeed_deg_s,
        "duration_s": command.duration_s,
    }
    if command.source_action is not None:
        payload["source_action"] = command.source_action.value
    if command.metadata:
        payload["metadata"] = command.metadata
    return payload


def _command_from_dict(value: dict[str, Any]) -> DroneCommand:
    source_action = value.get("source_action")
    return DroneCommand(
        forward_m_s=float(value.get("forward_m_s", 0.0)),
        right_m_s=float(value.get("right_m_s", 0.0)),
        down_m_s=float(value.get("down_m_s", 0.0)),
        yawspeed_deg_s=float(value.get("yawspeed_deg_s", 0.0)),
        duration_s=float(value.get("duration_s", 0.5)),
        source_action=coerce_discrete_action(source_action) if source_action else None,
        metadata=dict(value.get("metadata", {})),
    )


def _pose_to_dict(value: Pose | None) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = asdict(value)
    payload["position_xyz"] = list(value.position_xyz)
    payload["orientation_xyzw"] = list(value.orientation_xyzw)
    return payload


def _intrinsics_to_dict(value: CameraIntrinsics | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return asdict(value)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _resolve_record_path(value: str | None, episode_dir: Path) -> str | None:
    if value is None:
        return None

    path = Path(value)
    if not path.is_absolute():
        candidate = (episode_dir / path).resolve()
        return str(candidate)

    if path.exists():
        return str(path)

    workspace_prefix = Path("/workspace")
    try:
        relative = path.relative_to(workspace_prefix)
    except ValueError:
        return str(path)

    candidate = (PROJECT_ROOT / relative).resolve()
    if candidate.exists():
        return str(candidate)
    return str(path)


def _episode_hash_fraction(episode_id: str) -> float:
    digest = hashlib.md5(episode_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _step_matches_filters(
    step: BCEpisodeStep,
    *,
    include_worlds: set[str] | None,
    required_decision_family: str | None,
    allowed_decision_families: set[str] | None,
    require_decision_rich: bool,
    allowed_actions: set[str] | None,
) -> bool:
    metadata = dict(step.metadata or {})
    teacher = dict(metadata.get("teacher", {}))
    decision_profile = dict(teacher.get("decision_profile", {}))

    if include_worlds:
        world_label = metadata.get("world_label")
        if not world_label or str(world_label) not in include_worlds:
            return False

    if required_decision_family:
        family = decision_profile.get("family")
        if str(family) != str(required_decision_family):
            return False

    if allowed_decision_families:
        family = decision_profile.get("family")
        if str(family) not in allowed_decision_families:
            return False

    if require_decision_rich and not bool(decision_profile.get("decision_rich", False)):
        return False

    if allowed_actions and step.action.value not in allowed_actions:
        return False

    return True


def _step_decision_family(step: BCEpisodeStep) -> str:
    metadata = dict(step.metadata or {})
    teacher = dict(metadata.get("teacher", {}))
    decision_profile = dict(teacher.get("decision_profile", {}))
    return str(decision_profile.get("family") or "")


def _select_episode_steps(
    steps: list[BCEpisodeStep],
    *,
    include_worlds: set[str] | None,
    required_decision_family: str | None,
    allowed_decision_families: set[str] | None,
    require_decision_rich: bool,
    allowed_actions: set[str] | None,
    followthrough_after_families: set[str] | None,
    followthrough_family: str | None,
    followthrough_actions: set[str] | None,
    followthrough_steps: int,
) -> list[tuple[BCEpisodeStep, str]]:
    selected: list[tuple[BCEpisodeStep, str]] = []
    trigger_index: int | None = None
    max_followthrough_steps = int(max(0, followthrough_steps))

    for step_index, step in enumerate(steps):
        family = _step_decision_family(step)
        matches_base = _step_matches_filters(
            step,
            include_worlds=include_worlds,
            required_decision_family=required_decision_family,
            allowed_decision_families=allowed_decision_families,
            require_decision_rich=require_decision_rich,
            allowed_actions=allowed_actions,
        )

        if matches_base:
            selected.append((step, "base"))
        elif (
            trigger_index is not None
            and max_followthrough_steps > 0
            and (step_index - trigger_index) <= max_followthrough_steps
            and (not followthrough_family or family == followthrough_family)
            and (not followthrough_actions or step.action.value in followthrough_actions)
        ):
            selected.append((step, "followthrough"))

        if followthrough_after_families and family in followthrough_after_families:
            trigger_index = step_index
        elif trigger_index is not None and (step_index - trigger_index) >= max_followthrough_steps:
            trigger_index = None

    return selected
