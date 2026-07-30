#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_SUMMARY = PROJECT_ROOT / "data" / "bc_sft" / "manifests" / "bc_manifest_summary.json"
DEFAULT_BASELINE_SUMMARY = PROJECT_ROOT / "output" / "bc_policy_baseline" / "train_summary.json"
DEFAULT_PROGRESS_REPORT = PROJECT_ROOT / "output" / "bc_progress_report" / "index.html"
DEFAULT_BEHAVIOR_REPORT = PROJECT_ROOT / "output" / "episode_behavior_report" / "index.html"
DEFAULT_EVAL_REPORT = PROJECT_ROOT / "output" / "closed_loop_eval_report" / "index.html"
DEFAULT_EVAL_SUMMARY = PROJECT_ROOT / "output" / "closed_loop_eval_report" / "dashboard_summary.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "training_dashboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-summary", default=str(DEFAULT_MANIFEST_SUMMARY))
    parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    parser.add_argument("--progress-report", default=str(DEFAULT_PROGRESS_REPORT))
    parser.add_argument("--behavior-report", default=str(DEFAULT_BEHAVIOR_REPORT))
    parser.add_argument("--eval-report", default=str(DEFAULT_EVAL_REPORT))
    parser.add_argument("--eval-summary", default=str(DEFAULT_EVAL_SUMMARY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_summary = _load_json_if_exists(Path(args.manifest_summary).expanduser())
    baseline_summary = _load_json_if_exists(Path(args.baseline_summary).expanduser())
    eval_summary = _load_json_if_exists(Path(args.eval_summary).expanduser())

    report_path = out_dir / "index.html"
    report_path.write_text(
        build_html(
            out_dir=out_dir,
            manifest_summary=manifest_summary,
            baseline_summary=baseline_summary,
            eval_summary=eval_summary,
            progress_report=Path(args.progress_report).expanduser(),
            behavior_report=Path(args.behavior_report).expanduser(),
            eval_report=Path(args.eval_report).expanduser(),
        ),
        encoding="utf-8",
    )
    print(json.dumps({"report_path": str(report_path)}, indent=2))
    return 0


def build_html(
    *,
    out_dir: Path,
    manifest_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    eval_summary: dict[str, Any],
    progress_report: Path,
    behavior_report: Path,
    eval_report: Path,
) -> str:
    latest_val_acc = _latest_val_accuracy(baseline_summary)
    best_val_loss = baseline_summary.get("best_metric")
    eval_task = str(eval_summary.get("task_label", "waypoint"))
    if eval_task == "tree_avoidance":
        eval_success_label = "Escape Success"
        eval_progress_label = "Safe Continuation"
        eval_net_label = "Mean Probe Progress"
        eval_report_desc = "Whether checkpoints escaped blocked tree scenes, completed a safe forward continuation, or stalled in SITL."
    else:
        eval_success_label = "Eval Success"
        eval_progress_label = "Goal Progress"
        eval_net_label = "Mean Net Progress"
        eval_report_desc = "Whether checkpoints moved toward goals, stalled, or oscillated in SITL."
    sections = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>Training Dashboard</title>",
        "<style>",
        _style_block(),
        "</style>",
        "</head>",
        "<body><main>",
        "<div class='hero'>",
        "<div>",
        "<h1>Training Dashboard</h1>",
        "<p class='muted'>One place to monitor data scale, supervised training, teacher behavior, and live closed-loop checkpoint behavior.</p>",
        "</div>",
        "<div class='hero-callout'>",
        f"<div><strong>Latest Checkpoint:</strong> {html.escape(Path(str(baseline_summary.get('best_checkpoint', 'n/a'))).name)}</div>",
        f"<div><strong>Latest Eval Run:</strong> {html.escape(str(eval_summary.get('latest_run_id') or 'none'))}</div>",
        "</div>",
        "</div>",
        "<section class='cards'>",
        _metric_card("Episodes", str(manifest_summary.get("num_episodes", 0))),
        _metric_card("Examples", str(manifest_summary.get("num_examples", 0))),
        _metric_card("Train / Val", f"{manifest_summary.get('train_examples', 0)} / {manifest_summary.get('val_examples', 0)}"),
        _metric_card("Val Accuracy", _format_percent(latest_val_acc)),
        _metric_card("Best Val Loss", _format_float(best_val_loss)),
        _metric_card("Worlds", str(len(dict(manifest_summary.get("counts_by_world", {}))))),
        _metric_card(eval_success_label, _format_percent(eval_summary.get("success_rate"))),
        _metric_card(eval_progress_label, _format_percent(eval_summary.get("moved_toward_goal_rate"))),
        _metric_card(eval_net_label, f"{_format_float(eval_summary.get('mean_net_progress_m'))} m"),
        _metric_card("Eval Stall Rate", _format_percent(eval_summary.get("stall_rate"))),
        _metric_card("Eval Oscillation", _format_percent(eval_summary.get("mean_oscillation_rate"))),
        _metric_card("Decision-Rich", _format_percent(manifest_summary.get("decision_rich_fraction"))),
        _metric_card("Branch Score", _format_float(manifest_summary.get("mean_branch_score"))),
        _metric_card("Front Clearance +", _format_percent(eval_summary.get("front_clearance_improved_rate"))),
        _metric_card("Front Delta", f"{_format_float(eval_summary.get('mean_front_clearance_delta_m'))} m"),
        "</section>",
        "<section class='grid-two'>",
        "<article class='panel'>",
        "<h2>Training Snapshot</h2>",
        _key_value_table(
            [
                ("Action classes", str(len(manifest_summary.get("action_vocab", [])))),
                ("Validation episodes", str(manifest_summary.get("val_episode_count", 0))),
                ("Checkpoint device", str(baseline_summary.get("device", "n/a"))),
                ("Train examples seen", str(baseline_summary.get("num_train", 0))),
                ("Validation examples", str(baseline_summary.get("num_val", 0))),
                ("Decision-rich examples", str(manifest_summary.get("decision_rich_examples", 0))),
                ("World distribution", ", ".join(f"{k}:{v}" for k, v in sorted(dict(manifest_summary.get("counts_by_world", {})).items())) or "n/a"),
            ]
        ),
        "</article>",
        "<article class='panel'>",
        "<h2>Closed-Loop Snapshot</h2>",
        _key_value_table(
            [
                ("Eval runs", str(eval_summary.get("num_runs", 0))),
                ("Eval episodes", str(eval_summary.get("num_episodes", 0))),
                (eval_progress_label, _format_percent(eval_summary.get("moved_toward_goal_rate"))),
                ("Safety override rate", _format_percent(eval_summary.get("safety_override_rate"))),
                ("Front clearance improved", _format_percent(eval_summary.get("front_clearance_improved_rate"))),
                ("Mean front clearance delta", f"{_format_float(eval_summary.get('mean_front_clearance_delta_m'))} m"),
                ("Latest eval checkpoint", Path(str(eval_summary.get("latest_checkpoint_path") or "n/a")).name),
            ]
        ),
        "</article>",
        "</section>",
        "<section class='links'>",
        "<h2>Detailed Views</h2>",
        _report_link_card(out_dir, progress_report, "BC Progress", "Dataset growth, cycle trends, action mix, and recent samples."),
        _report_link_card(out_dir, behavior_report, "Episode Behavior", "Teacher reason mix and what the collector actually did by episode and run."),
        _report_link_card(out_dir, eval_report, "Closed-Loop Eval", eval_report_desc),
        "</section>",
        "<section class='panel'>",
        "<h2>Action Distribution</h2>",
        _action_table(dict(manifest_summary.get("counts_by_action", {}))),
        "</section>",
        "</main></body></html>",
    ]
    return "\n".join(sections)


def _report_link_card(out_dir: Path, target: Path, label: str, desc: str) -> str:
    href = _relative_asset(out_dir, target) if target.exists() else "#"
    state = "available" if target.exists() else "missing"
    return (
        "<article class='panel link-card'>"
        f"<h3>{html.escape(label)}</h3>"
        f"<p>{html.escape(desc)}</p>"
        f"<p class='small'>Status: {html.escape(state)}</p>"
        f"<a href='{html.escape(href)}'>Open report</a>"
        "</article>"
    )


def _key_value_table(rows: list[tuple[str, str]]) -> str:
    parts = ["<table><tbody>"]
    for label, value in rows:
        parts.append(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f"<td>{html.escape(value)}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _action_table(counts: dict[str, Any]) -> str:
    if not counts:
        return "<p class='muted'>No action counts found.</p>"
    total = sum(int(value) for value in counts.values())
    parts = ["<table><thead><tr><th>Action</th><th>Count</th><th>Share</th></tr></thead><tbody>"]
    for action, count in sorted(counts.items(), key=lambda item: (-int(item[1]), item[0])):
        share = int(count) / max(total, 1)
        parts.append(
            "<tr>"
            f"<td>{html.escape(action)}</td>"
            f"<td>{int(count)}</td>"
            f"<td>{html.escape(_format_percent(share))}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_val_accuracy(summary: dict[str, Any]) -> float | None:
    history = list(summary.get("history", []))
    values = [
        float(epoch["val"]["accuracy"])
        for epoch in history
        if isinstance(epoch, dict) and epoch.get("val") is not None
    ]
    if not values:
        return None
    return values[-1]


def _relative_asset(out_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=out_dir.resolve())


def _metric_card(label: str, value: str) -> str:
    return (
        "<article class='card'>"
        f"<div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div>"
        "</article>"
    )


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _style_block() -> str:
    return """
    :root {
      --bg: #0d1117;
      --panel: #101925;
      --panel-2: #162332;
      --text: #f8fafc;
      --muted: #94a3b8;
      --line: #253446;
      --accent: #22d3ee;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(34, 211, 238, 0.12), transparent 28%),
        radial-gradient(circle at bottom right, rgba(16, 185, 129, 0.1), transparent 24%),
        linear-gradient(180deg, #0d1117, #0f1720 45%, #0d1117);
    }
    main { max-width: 1400px; margin: 0 auto; padding: 28px; }
    .hero {
      display: flex;
      gap: 20px;
      justify-content: space-between;
      align-items: flex-start;
    }
    .hero-callout {
      min-width: 320px;
      background: rgba(16, 25, 37, 0.85);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
    }
    h1, h2, h3 { margin: 0 0 12px 0; }
    p { line-height: 1.5; }
    .muted, .small { color: var(--muted); }
    .small { font-size: 0.9rem; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 24px 0;
    }
    .card, .panel {
      background: rgba(16, 25, 37, 0.88);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.26);
    }
    .label { color: var(--muted); font-size: 0.85rem; }
    .value { font-size: 1.7rem; margin-top: 6px; }
    .grid-two {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }
    .links {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }
    .link-card a { color: var(--accent); text-decoration: none; }
    .link-card a:hover { text-decoration: underline; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: rgba(16, 25, 37, 0.35);
      border-radius: 12px;
      overflow: hidden;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid rgba(37, 52, 70, 0.7);
      text-align: left;
    }
    th { color: var(--muted); width: 45%; }
    """


if __name__ == "__main__":
    raise SystemExit(main())
