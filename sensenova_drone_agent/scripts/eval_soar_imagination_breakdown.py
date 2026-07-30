#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from residual_adapter_runtime import infer_adapter_action_overrides, wrap_dynamics_with_residual_adapter  # noqa: E402
from train_native_dreamer4_imagination import (  # noqa: E402
    DEFAULT_ACTION_DIM,
    AgentHeads,
    NativeImaginationConfig,
    apply_episode_holdout_split,
    build_balanced_eval_indices,
    build_dynamics,
    evaluate_policies,
    load_frozen_tokenizer_from_pt_ckpt,
    policy_single_action_dim,
    prepare_batch,
    rollout_return,
    set_eval_seed,
)
from wm_dataset import WMDataset, collate_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SOAR/DROID imagination policy by source, horizon, and action controls.")
    parser.add_argument("--run-dir", required=True, help="Imagination run directory containing after_imagination.pt and summary/config.")
    parser.add_argument("--checkpoint", default="after_imagination.pt")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--sources", default="all,soar,droid")
    parser.add_argument("--horizons", default="4,8,16")
    parser.add_argument("--controls", default="policy,zero,shuffle,time_shift,far_shuffle")
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    out_dir = resolve_path(args.out_dir) if args.out_dir else run_dir / "breakdown_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(resolve_path(run_dir / args.checkpoint), map_location="cpu", weights_only=False)
    config = NativeImaginationConfig(**checkpoint["config"])
    config.device = str(device)
    config.eval_batches = int(args.eval_batches)
    config.batch_size = int(args.batch_size)
    config.num_workers = int(args.num_workers)
    if args.eval_seed:
        config.eval_seed = int(args.eval_seed)
    config.eval_causal_dynamics = True

    tokenizer_ckpt = resolve_path(config.tokenizer_ckpt)
    dynamics_ckpt = resolve_path(config.dynamics_ckpt)
    dyn_ckpt = torch.load(dynamics_ckpt, map_location="cpu", weights_only=False)
    dyn_args = dict(dyn_ckpt.get("args", {}))
    adapter_ckpt = resolve_path(config.residual_adapter_ckpt) if config.residual_adapter_ckpt else None
    adapter_action_overrides = infer_adapter_action_overrides(adapter_ckpt)
    dyn_args["action_dim"] = int(adapter_action_overrides.get("action_dim", config.action_dim or DEFAULT_ACTION_DIM))
    dyn_args["action_features"] = str(adapter_action_overrides.get("action_features", config.action_features))

    encoder, _decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(str(tokenizer_ckpt), device=device)
    dynamics = build_dynamics(dyn_args, tok_args).to(device)
    dynamics.load_state_dict(dyn_ckpt["dynamics"], strict=True)
    dynamics.eval()
    for param in dynamics.parameters():
        param.requires_grad_(False)
    if adapter_ckpt is not None:
        dynamics, _adapter_info = wrap_dynamics_with_residual_adapter(
            base=dynamics,
            adapter_ckpt=adapter_ckpt,
            dyn_args=dyn_args,
            tok_args=tok_args,
            device=device,
        )

    z_dim = int(tok_args.get("n_latents", 16) // int(dyn_args.get("packing_factor", 2))) * int(
        tok_args.get("d_bottleneck", 32)
    ) * int(dyn_args.get("packing_factor", 2))
    policy_dim = policy_single_action_dim(config)
    agent = AgentHeads(
        2 * z_dim + policy_dim,
        policy_dim * int(config.action_chunk_len),
        int(config.hidden_dim),
        float(config.log_std_init),
        z_dim=z_dim,
        single_action_dim=policy_dim,
    ).to(device)
    agent.load_state_dict(checkpoint["agent"], strict=True)
    agent.eval()

    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    horizons = [int(item.strip()) for item in args.horizons.split(",") if item.strip()]
    controls = [item.strip() for item in args.controls.split(",") if item.strip()]
    rows = []
    per_source: dict[str, Any] = {}
    for source in sources:
        source_config = copy.deepcopy(config)
        source_config.data_dirs, source_config.frame_dirs = select_source_paths(config, source)
        if not source_config.data_dirs:
            continue
        loader, task_names, split_info, eval_sampling_info = build_eval_loader(source_config)
        per_source[source] = {"split": split_info, "eval_sampling": eval_sampling_info}
        for horizon in horizons:
            eval_config = copy.deepcopy(source_config)
            eval_config.imagination_horizon = int(horizon)
            eval_config.eval_batches = int(args.eval_batches)
            eval_config.batch_size = int(args.batch_size)
            eval_config.num_workers = int(args.num_workers)
            set_eval_seed(eval_config)
            base = evaluate_policies(
                agent=agent,
                encoder=encoder,
                dynamics=dynamics,
                loader=loader,
                tok_args=tok_args,
                dyn_args=dyn_args,
                config=eval_config,
                device=device,
                task_names=task_names,
            )
            control_returns = {}
            for control in controls:
                if control == "policy":
                    control_returns[control] = float(base["policy"])
                elif control == "zero":
                    control_returns[control] = float(base["policy_dyn_zero"])
                elif control == "shuffle":
                    control_returns[control] = float(base["policy_dyn_shuffle"])
                else:
                    control_returns[control] = evaluate_policy_control_return(
                        agent=agent,
                        encoder=encoder,
                        dynamics=dynamics,
                        loader=loader,
                        tok_args=tok_args,
                        dyn_args=dyn_args,
                        config=eval_config,
                        device=device,
                        dynamics_action_mode=control,
                    )
            row = {
                "source": source,
                "horizon": int(horizon),
                "zero": float(base["zero"]),
                "bc_prior": float(base["bc_prior"]),
                "policy": float(base["policy"]),
                "policy_minus_bc": float(base["policy_minus_bc"]),
                "policy_minus_zero": float(base["policy_minus_zero"]),
                "policy_minus_dyn_zero": float(base["policy_minus_dyn_zero"]),
                "policy_minus_dyn_shuffle": float(base["policy_minus_dyn_shuffle"]),
                "causal_policy_gain": float(base["causal_policy_gain"]),
                "controls": control_returns,
                "control_margins": {
                    name: float(base["policy"]) - value for name, value in control_returns.items() if name != "policy"
                },
                "per_task": base.get("per_task", {}),
            }
            rows.append(row)
            print(json.dumps({"phase": "breakdown_eval", **{k: v for k, v in row.items() if k not in {"per_task"}}}), flush=True)

    summary = {
        "phase": "soar_imagination_breakdown_eval",
        "run_dir": str(run_dir),
        "checkpoint": str(resolve_path(run_dir / args.checkpoint)),
        "config": asdict(config),
        "sources": per_source,
        "rows": rows,
        "claim_boundary": "Breakdown is learned-simulator evaluation only; it does not prove real-world robot control.",
    }
    (out_dir / "breakdown_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out_dir / "breakdown_report.md")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, indent=2))
    return 0


def build_eval_loader(config: NativeImaginationConfig) -> tuple[DataLoader, list[str], dict[str, Any], dict[str, Any]]:
    dataset = WMDataset(
        data_dir=config.data_dirs,
        frames_dir=config.frame_dirs,
        seq_len=config.seq_len,
        img_size=128,
        action_dim=config.action_dim,
        raw_action_dim=config.raw_action_dim,
        tasks_json=config.tasks_json,
        tasks=None,
        strict_tasks=False,
        action_features=config.action_features,
        require_non_noop=config.require_non_noop,
        no_op_threshold=config.no_op_threshold,
        min_non_noop_steps=config.min_non_noop_steps,
        reward_filter_mode=config.reward_filter_mode,
        reward_signal_threshold=config.reward_signal_threshold,
        min_reward_signal_steps=config.min_reward_signal_steps,
        verbose=False,
    )
    train_shadow = WMDataset(
        data_dir=config.data_dirs,
        frames_dir=config.frame_dirs,
        seq_len=config.seq_len,
        img_size=128,
        action_dim=config.action_dim,
        raw_action_dim=config.raw_action_dim,
        tasks_json=config.tasks_json,
        tasks=None,
        strict_tasks=False,
        action_features=config.action_features,
        verbose=False,
    )
    split_info = apply_episode_holdout_split(
        train_shadow,
        dataset,
        holdout_fraction=config.eval_holdout_fraction,
        seed=config.split_seed,
    )
    eval_indices, eval_sampling_info = build_balanced_eval_indices(
        dataset,
        num_batches=config.eval_batches,
        batch_size=config.batch_size,
        seed=config.eval_seed if config.eval_seed else config.seed + 15485863,
    )
    loader = DataLoader(
        Subset(dataset, eval_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_batch,
        persistent_workers=config.num_workers > 0,
    )
    return loader, dataset.tasks, split_info, eval_sampling_info


@torch.no_grad()
def evaluate_policy_control_return(
    *,
    agent: AgentHeads,
    encoder: Any,
    dynamics: Any,
    loader: Any,
    tok_args: dict[str, Any],
    dyn_args: dict[str, Any],
    config: NativeImaginationConfig,
    device: torch.device,
    dynamics_action_mode: str,
) -> float:
    vals: list[float] = []
    for i, raw_batch in enumerate(loader):
        if i >= config.eval_batches:
            break
        batch = prepare_batch(raw_batch, encoder, tok_args, dyn_args, config, device)
        ret = rollout_return(agent, dynamics, batch, dyn_args, config, mode="policy", dynamics_action_mode=dynamics_action_mode)
        vals.append(float(ret.mean().item()))
    return sum(vals) / max(1, len(vals))


def select_source_paths(config: NativeImaginationConfig, source: str) -> tuple[list[str], list[str]]:
    if source == "all":
        return list(config.data_dirs), list(config.frame_dirs)
    selected_data = []
    selected_frames = []
    for data_dir, frame_dir in zip(config.data_dirs, config.frame_dirs):
        text = f"{data_dir} {frame_dir}".lower()
        if source == "soar" and "robotics/soar" in text:
            selected_data.append(data_dir)
            selected_frames.append(frame_dir)
        elif source == "droid" and ("droid" in text or "hf_action_exports" in text):
            selected_data.append(data_dir)
            selected_frames.append(frame_dir)
    return selected_data, selected_frames


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# SOAR/DROID Imagination Breakdown",
        "",
        f"Run: `{summary['run_dir']}`",
        "",
        "| Source | Horizon | Policy - BC | Policy - zero | Policy - dyn-zero | Policy - dyn-shuffle | Time-shift margin | Far-shuffle margin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        margins = row.get("control_margins", {})
        lines.append(
            "| {source} | {horizon} | {bc:+.4f} | {zero:+.4f} | {dyn_zero:+.4f} | {dyn_shuffle:+.4f} | {time_shift:+.4f} | {far_shuffle:+.4f} |".format(
                source=row["source"],
                horizon=row["horizon"],
                bc=row["policy_minus_bc"],
                zero=row["policy_minus_zero"],
                dyn_zero=row["policy_minus_dyn_zero"],
                dyn_shuffle=row["policy_minus_dyn_shuffle"],
                time_shift=float(margins.get("time_shift", 0.0)),
                far_shuffle=float(margins.get("far_shuffle", 0.0)),
            )
        )
    lines.extend(["", summary["claim_boundary"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    text = str(value)
    if text.startswith("/workspace/"):
        return (REPO_ROOT / text[len("/workspace/") :]).resolve()
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
