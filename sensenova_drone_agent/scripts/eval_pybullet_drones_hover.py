#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


TARGET_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate simple policies on gym-pybullet-drones HoverAviary."
    )
    parser.add_argument("--out-dir", default="sensenova_drone_agent/output/pybullet_drones_hover_v1")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=120000)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--obs", choices=["rgb", "kin"], default="rgb")
    parser.add_argument("--act", choices=["vel"], default="vel")
    parser.add_argument("--policies", default="target_velocity,random,zero_velocity")
    parser.add_argument("--initial-z", type=float, default=0.2)
    parser.add_argument("--success-distance-m", type=float, default=0.15)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--save-contact-sheet", action="store_true", default=True)
    parser.add_argument("--trace-frames", type=int, default=8)
    parser.add_argument(
        "--kairos-vae-probe",
        action="store_true",
        help="Extract native Kairos/Wan VAE features from RGB frames while target_velocity controls.",
    )
    parser.add_argument("--kairos-probe-every-n", type=int, default=24)
    parser.add_argument("--kairos-device", default="cpu")
    parser.add_argument("--kairos-dtype", default="float32")
    parser.add_argument("--kairos-height", type=int, default=128)
    parser.add_argument("--kairos-width", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imports = import_pybullet_drones()
    policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]
    if args.kairos_vae_probe and "kairos_vae_probe" not in policy_names:
        policy_names.append("kairos_vae_probe")

    summary: dict[str, Any] = {
        "benchmark": "gym-pybullet-drones HoverAviary",
        "package": package_metadata(),
        "episodes": args.episodes,
        "seed_start": args.seed,
        "obs": args.obs,
        "act": args.act,
        "target_pos": TARGET_POS.tolist(),
        "success_distance_m": args.success_distance_m,
        "policies": [],
    }

    all_results = []
    for policy_name in policy_names:
        result = evaluate_policy(policy_name, args, imports, out_dir)
        all_results.append(result)
        summary["policies"].append(summarize_policy(result))

    summary["ranking_by_success_then_distance"] = sorted(
        [
            {
                "policy": item["policy"],
                "success_rate": item["success_rate"],
                "mean_final_distance_m": item["mean_final_distance_m"],
                "mean_return": item["mean_return"],
            }
            for item in summary["policies"]
        ],
        key=lambda row: (-row["success_rate"], row["mean_final_distance_m"]),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, out_dir / "report.md")
    print(json.dumps(summary["ranking_by_success_then_distance"], indent=2))
    print(f"Wrote {out_dir / 'summary.json'}")
    return 0


def import_pybullet_drones() -> dict[str, Any]:
    try:
        from gym_pybullet_drones.envs.HoverAviary import HoverAviary
        from gym_pybullet_drones.utils.enums import ActionType, ObservationType, Physics
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"gym-pybullet-drones or one of its runtime dependencies is missing: {exc}. "
            "Build the benchmark image with "
            "`./sensenova_drone_agent/scripts/build_pybullet_drones_benchmark_image.sh` "
            "and run via `./sensenova_drone_agent/scripts/run_pybullet_drones_benchmark.sh`."
        ) from exc
    return {
        "HoverAviary": HoverAviary,
        "ActionType": ActionType,
        "ObservationType": ObservationType,
        "Physics": Physics,
    }


def package_metadata() -> dict[str, Any]:
    try:
        import gym_pybullet_drones

        version = getattr(gym_pybullet_drones, "__version__", None)
    except Exception:
        version = None
    return {
        "name": "gym-pybullet-drones",
        "version": version,
        "source": "https://github.com/learnsyslab/gym-pybullet-drones",
    }


def evaluate_policy(policy_name: str, args: argparse.Namespace, imports: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    policy_dir = out_dir / policy_name
    policy_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = policy_dir / "episodes.jsonl"
    traces_dir = policy_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    extractor = None
    feature_records = []
    if policy_name == "kairos_vae_probe":
        from sensenova_drone.kairos_features import KairosVAEFeatureExtractor

        extractor = KairosVAEFeatureExtractor(
            repo_root=REPO_ROOT,
            device=args.kairos_device,
            dtype=args.kairos_dtype,
            height=args.kairos_height,
            width=args.kairos_width,
            tiled=False,
        )

    records = []
    with episodes_path.open("w", encoding="utf-8") as f:
        for episode_idx in range(args.episodes):
            seed = args.seed + episode_idx
            record = run_episode(
                policy_name,
                seed,
                args,
                imports,
                traces_dir=traces_dir,
                extractor=extractor,
            )
            records.append(record)
            f.write(json.dumps(record) + "\n")
            for item in record.get("kairos_feature_records", []):
                feature_records.append(item)

    if feature_records:
        (policy_dir / "kairos_feature_records.json").write_text(
            json.dumps(feature_records, indent=2), encoding="utf-8"
        )

    result = {
        "policy": policy_name,
        "episodes_path": str(episodes_path),
        "records": records,
        "kairos_feature_records_path": str(policy_dir / "kairos_feature_records.json") if feature_records else None,
    }
    if args.save_contact_sheet:
        trace_paths = [
            Path(record["trace_contact_sheet"])
            for record in records
            if record.get("trace_contact_sheet")
        ]
        if trace_paths:
            combined = policy_dir / "contact_sheet.png"
            make_combined_contact_sheet(trace_paths[: min(len(trace_paths), 4)], combined)
            result["contact_sheet"] = str(combined)
    return result


def run_episode(
    policy_name: str,
    seed: int,
    args: argparse.Namespace,
    imports: dict[str, Any],
    traces_dir: Path,
    extractor: Any | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    env = make_env(args, imports)
    frames = []
    feature_records = []
    total_reward = 0.0
    min_distance = float("inf")
    final_distance = float("inf")
    terminated = False
    truncated = False
    start_time = time.time()
    try:
        obs, info = env.reset(seed=seed)
        del info
        for step in range(args.max_steps):
            state = env._getDroneStateVector(0)
            distance = float(np.linalg.norm(TARGET_POS - state[0:3]))
            min_distance = min(min_distance, distance)
            final_distance = distance

            if should_capture_frame(step, args.trace_frames, args.max_steps):
                frames.append(rgb_from_obs(obs, args.obs))

            if extractor is not None and args.obs == "rgb" and step % max(1, args.kairos_probe_every_n) == 0:
                feature_records.append(extract_feature_record(extractor, obs, step))

            action = choose_action(policy_name, env, state)
            obs, reward, terminated, truncated, info = env.step(action)
            del info
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()

    success = final_distance <= args.success_distance_m
    trace_path = None
    if frames:
        trace_path = traces_dir / f"{policy_name}_{seed}.png"
        make_frame_contact_sheet(frames, trace_path, label=f"{policy_name} seed={seed}")

    return {
        "policy": policy_name,
        "seed": seed,
        "steps": step + 1 if "step" in locals() else 0,
        "return": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(success),
        "final_distance_m": final_distance,
        "min_distance_m": min_distance,
        "elapsed_s": time.time() - start_time,
        "trace_contact_sheet": str(trace_path) if trace_path else None,
        "kairos_feature_records": feature_records,
    }


def make_env(args: argparse.Namespace, imports: dict[str, Any]):
    HoverAviary = imports["HoverAviary"]
    ObservationType = imports["ObservationType"]
    ActionType = imports["ActionType"]
    Physics = imports["Physics"]

    obs_type = ObservationType.RGB if args.obs == "rgb" else ObservationType.KIN
    act_type = ActionType.VEL
    initial_xyzs = np.array([[0.0, 0.0, args.initial_z]], dtype=np.float32)
    return HoverAviary(
        initial_xyzs=initial_xyzs,
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=24 if args.obs == "rgb" else 30,
        gui=args.gui,
        record=args.record,
        obs=obs_type,
        act=act_type,
    )


def choose_action(policy_name: str, env: Any, state: np.ndarray) -> np.ndarray:
    if policy_name == "random":
        return env.action_space.sample()

    if policy_name in {"zero_velocity", "hover"}:
        return np.zeros((1, 4), dtype=np.float32)

    if policy_name in {"target_velocity", "kairos_vae_probe"}:
        delta = TARGET_POS - state[0:3]
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            direction = np.zeros(3, dtype=np.float32)
        else:
            direction = (delta / norm).astype(np.float32)
        speed_fraction = np.float32(np.clip(norm, 0.1, 1.0))
        return np.array([[direction[0], direction[1], direction[2], speed_fraction]], dtype=np.float32)

    raise ValueError(f"Unknown policy: {policy_name}")


def extract_feature_record(extractor: Any, obs: np.ndarray, step: int) -> dict[str, Any]:
    frame = rgb_from_obs(obs, "rgb")
    payload = extractor.encode_image(frame)
    features = payload["image_features"]
    return {
        "step": step,
        "backend": payload["metadata"]["backend"],
        "latent_shape": payload["metadata"]["latent_shape"],
        "feature_dim": payload["metadata"]["feature_dim"],
        "feature_mean": float(features.mean().item()),
        "feature_std": float(features.std(unbiased=False).item()),
    }


def rgb_from_obs(obs: np.ndarray, obs_type: str) -> np.ndarray | None:
    if obs_type != "rgb":
        return None
    frame = np.asarray(obs[0])
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.clip(frame, 0, 255).astype(np.uint8)


def should_capture_frame(step: int, trace_frames: int, max_steps: int) -> bool:
    if trace_frames <= 0:
        return False
    interval = max(1, max_steps // trace_frames)
    return step % interval == 0


def summarize_policy(result: dict[str, Any]) -> dict[str, Any]:
    records = result["records"]
    n = max(1, len(records))
    return {
        "policy": result["policy"],
        "episodes": len(records),
        "success_rate": sum(1 for record in records if record["success"]) / n,
        "terminated_rate": sum(1 for record in records if record["terminated"]) / n,
        "truncated_rate": sum(1 for record in records if record["truncated"]) / n,
        "mean_return": float(np.mean([record["return"] for record in records])),
        "mean_length": float(np.mean([record["steps"] for record in records])),
        "mean_final_distance_m": float(np.mean([record["final_distance_m"] for record in records])),
        "mean_min_distance_m": float(np.mean([record["min_distance_m"] for record in records])),
        "episodes_path": result["episodes_path"],
        "contact_sheet": result.get("contact_sheet"),
        "kairos_feature_records_path": result.get("kairos_feature_records_path"),
    }


def make_frame_contact_sheet(frames: list[np.ndarray | None], out_path: Path, label: str = "") -> None:
    valid = [frame for frame in frames if frame is not None]
    if not valid:
        return
    pil_frames = [Image.fromarray(frame).resize((160, 120)) for frame in valid]
    label_h = 24
    sheet = Image.new("RGB", (160 * len(pil_frames), 120 + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), label, fill=(0, 0, 0))
    for idx, frame in enumerate(pil_frames):
        sheet.paste(frame, (idx * 160, label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def make_combined_contact_sheet(paths: list[Path], out_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths if path.exists()]
    if not images:
        return
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(out_path)


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# gym-pybullet-drones Hover Benchmark",
        "",
        "External benchmark source: https://github.com/learnsyslab/gym-pybullet-drones",
        "",
        "## Results",
        "",
        "| Policy | Episodes | Success | Truncated | Mean Return | Mean Final Distance | Mean Min Distance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["policies"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["policy"],
                    str(row["episodes"]),
                    f"{row['success_rate']:.3f}",
                    f"{row['truncated_rate']:.3f}",
                    f"{row['mean_return']:.3f}",
                    f"{row['mean_final_distance_m']:.3f}",
                    f"{row['mean_min_distance_m']:.3f}",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This is an external drone physics benchmark smoke path. The privileged `target_velocity` policy is a sanity baseline, not a learned vision policy.",
        "The `kairos_vae_probe` mode extracts native Kairos/Wan VAE features from RGB observations while the sanity controller flies.",
        "A paper claim requires training/evaluating a policy that actually uses those features for action selection.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_out_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
