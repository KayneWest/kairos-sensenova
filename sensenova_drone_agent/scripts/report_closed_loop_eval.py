#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensenova_drone.eval.closed_loop import (
    aggregate_closed_loop_summaries,
    load_eval_episode_summary,
)


DEFAULT_EVAL_ROOT = PROJECT_ROOT / "output" / "closed_loop_eval"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "closed_loop_eval_report"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--recent-runs", type=int, default=8)
    parser.add_argument("--recent-episodes", type=int, default=8)
    parser.add_argument("--run-id-prefix", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_root = Path(args.eval_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_eval_runs(eval_root, run_id_prefix=args.run_id_prefix.strip() or None)
    report_path = out_dir / "index.html"
    summary_path = out_dir / "dashboard_summary.json"
    summary_payload = build_dashboard_summary(runs)
    report_path.write_text(
        build_report_html(
            out_dir=out_dir,
            runs=runs[: max(args.recent_runs, 0)],
            recent_episodes=_collect_recent_episodes(runs, max_count=max(args.recent_episodes, 0)),
            summary_payload=summary_payload,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(report_path), "summary_path": str(summary_path)}, indent=2))
    return 0


def collect_eval_runs(eval_root: Path, run_id_prefix: str | None = None) -> list[dict[str, Any]]:
    if not eval_root.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for run_dir in sorted((path for path in eval_root.iterdir() if path.is_dir()), reverse=True):
        run_summary = _load_json_if_exists(run_dir / "summary.json")
        run_id = str(run_summary.get("run_id", run_dir.name))
        if run_id_prefix is not None and not run_id.startswith(run_id_prefix):
            continue
        episode_dirs = sorted((path for path in run_dir.iterdir() if path.is_dir() and (path / "episode.json").is_file()))
        episode_summaries = [load_eval_episode_summary(path) for path in episode_dirs]
        aggregate = aggregate_closed_loop_summaries(episode_summaries)
        runs.append(
            {
                "run_dir": run_dir,
                "run_id": run_id,
                "checkpoint_path": str(run_summary.get("checkpoint_path", "")),
                "created_utc": str(run_summary.get("created_utc", "")),
                "world_label": str(run_summary.get("world_label", "unknown")),
                "task_label": str(run_summary.get("task_label", aggregate.get("task_label", "waypoint"))),
                "aggregate": aggregate,
                "episodes": [asdict(summary) for summary in episode_summaries],
            }
        )
    return runs


def build_dashboard_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    recent_episodes = _collect_recent_episodes(runs, max_count=64)
    if not recent_episodes:
        return {
            "task_label": "waypoint",
            "num_runs": 0,
            "num_episodes": 0,
            "latest_run_id": None,
            "latest_checkpoint_path": None,
            "success_rate": 0.0,
            "moved_toward_goal_rate": 0.0,
            "stall_rate": 0.0,
            "mean_net_progress_m": 0.0,
            "mean_progress_ratio": 0.0,
            "mean_oscillation_rate": 0.0,
            "safety_override_rate": 0.0,
        }

    aggregates = aggregate_closed_loop_summaries(
        [
            _episode_summary_from_dict(episode)
            for episode in recent_episodes
        ]
    )
    latest_run = runs[0] if runs else {}
    return {
        "task_label": str(latest_run.get("task_label", aggregates.get("task_label", "waypoint"))),
        "num_runs": len(runs),
        "num_episodes": len(recent_episodes),
        "latest_run_id": latest_run.get("run_id"),
        "latest_checkpoint_path": latest_run.get("checkpoint_path"),
        **aggregates,
    }


def build_report_html(
    *,
    out_dir: Path,
    runs: list[dict[str, Any]],
    recent_episodes: list[dict[str, Any]],
    summary_payload: dict[str, Any],
) -> str:
    task_label = str(summary_payload.get("task_label", "waypoint"))
    if task_label == "tree_avoidance":
        intro = "Summarizes live SITL checkpoint runs and whether the policy escaped blocked tree scenes."
        success_label = "Escape Success"
        moved_label = "Safe Continuation"
        net_label = "Mean Probe Progress"
    else:
        intro = "Summarizes live SITL checkpoint runs and whether the policy actually moved toward sampled local goals."
        success_label = "Success Rate"
        moved_label = "Moved Toward Goal"
        net_label = "Mean Net Progress"
    sections = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>Closed-Loop Eval Report</title>",
        "<style>",
        _style_block(),
        "</style>",
        "</head>",
        "<body><main>",
        "<h1>Closed-Loop Eval Report</h1>",
        f"<p class='muted'>{html.escape(intro)}</p>",
        "<section class='cards'>",
        _metric_card("Eval Runs", str(summary_payload.get("num_runs", 0))),
        _metric_card("Episodes", str(summary_payload.get("num_episodes", 0))),
        _metric_card(success_label, _format_percent(summary_payload.get("success_rate"))),
        _metric_card(moved_label, _format_percent(summary_payload.get("moved_toward_goal_rate"))),
        _metric_card("Stall Rate", _format_percent(summary_payload.get("stall_rate"))),
        _metric_card(net_label, f"{_format_float(summary_payload.get('mean_net_progress_m'))} m"),
        _metric_card("Mean Oscillation", _format_percent(summary_payload.get("mean_oscillation_rate"))),
        _metric_card("Safety Override Rate", _format_percent(summary_payload.get("safety_override_rate"))),
        _metric_card("Front Clearance +", _format_percent(summary_payload.get("front_clearance_improved_rate"))),
        _metric_card("Mean Front Delta", f"{_format_float(summary_payload.get('mean_front_clearance_delta_m'))} m"),
        "</section>",
        "<section>",
        "<h2>Recent Runs</h2>",
        _run_table(runs),
        "</section>",
        "<section>",
        "<h2>Recent Episodes</h2>",
        _episode_gallery(out_dir, recent_episodes),
        "</section>",
        "</main></body></html>",
    ]
    return "\n".join(sections)


def _run_table(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "<p class='muted'>No closed-loop eval runs found.</p>"
    parts = [
        "<table><thead><tr><th>Run</th><th>World</th><th>Task</th><th>Checkpoint</th><th>Episodes</th><th>Success</th><th>Moved/Clearance</th><th>Stall</th><th>Net Metric</th><th>Front Delta</th><th>Oscillation</th></tr></thead><tbody>"
    ]
    for run in runs:
        aggregate = dict(run.get("aggregate", {}))
        parts.append(
            "<tr>"
            f"<td>{html.escape(str(run.get('run_id', 'unknown')))}</td>"
            f"<td>{html.escape(str(run.get('world_label', 'unknown')))}</td>"
            f"<td>{html.escape(str(run.get('task_label', 'waypoint')))}</td>"
            f"<td class='small'>{html.escape(Path(str(run.get('checkpoint_path', ''))).name)}</td>"
            f"<td>{int(aggregate.get('num_episodes', 0))}</td>"
            f"<td>{html.escape(_format_percent(aggregate.get('success_rate')))}</td>"
            f"<td>{html.escape(_format_percent(aggregate.get('moved_toward_goal_rate')))}</td>"
            f"<td>{html.escape(_format_percent(aggregate.get('stall_rate')))}</td>"
            f"<td>{html.escape(_format_float(aggregate.get('mean_net_progress_m')))} m</td>"
            f"<td>{html.escape(_format_float(aggregate.get('mean_front_clearance_delta_m')))} m</td>"
            f"<td>{html.escape(_format_percent(aggregate.get('mean_oscillation_rate')))}</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _episode_gallery(out_dir: Path, episodes: list[dict[str, Any]]) -> str:
    if not episodes:
        return "<p class='muted'>No episode eval artifacts found.</p>"
    parts = ["<div class='gallery'>"]
    for episode in episodes:
        task_label = str(episode.get("task_label", "waypoint"))
        net_label = "Probe Progress" if task_label == "tree_avoidance" else "Net Progress"
        moved_label = "Safe Continuation" if task_label == "tree_avoidance" else "Moved Toward Goal"
        parts.append("<article class='episode'>")
        parts.append(f"<h3>{html.escape(str(episode['episode_id']))}</h3>")
        parts.append(
            "<p>"
            f"<strong>Status:</strong> {html.escape(str(episode['status']))}<br/>"
            f"<strong>Steps:</strong> {int(episode['step_count'])}<br/>"
            f"<strong>{html.escape(net_label)}:</strong> {_format_float(episode.get('net_progress_m'))} m<br/>"
            f"<strong>Front Delta:</strong> {_format_float(episode.get('front_clearance_delta_m'))} m<br/>"
            f"<strong>{html.escape(moved_label)}:</strong> {bool(episode.get('moved_toward_goal'))}<br/>"
            f"<strong>Stalled:</strong> {bool(episode.get('stalled'))}<br/>"
            f"<strong>Oscillation:</strong> {_format_percent(episode.get('oscillation_rate'))}<br/>"
            f"<strong>Actions:</strong> {html.escape(', '.join(episode.get('actions', [])))}"
            "</p>"
        )
        contact_strip = episode.get("contact_strip_path")
        if contact_strip:
            rel = _relative_asset(out_dir, Path(str(contact_strip)))
            parts.append(f"<img src='{html.escape(rel)}' alt='contact strip'/>")
        parts.append(
            "<p class='small'>"
            f"<a href='{html.escape(_relative_asset(out_dir, Path(str(episode['episode_dir'])) / 'episode.json'))}'>episode.json</a>"
            "</p>"
        )
        parts.append("</article>")
    parts.append("</div>")
    return "\n".join(parts)


def _collect_recent_episodes(runs: list[dict[str, Any]], *, max_count: int) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for run in runs:
        for episode in run.get("episodes", []):
            episodes.append(episode)
    episodes.sort(key=lambda item: str(item.get("episode_id", "")), reverse=True)
    return episodes[:max_count]


def _episode_summary_from_dict(value: dict[str, Any]):
    from sensenova_drone.eval.closed_loop import ClosedLoopEpisodeSummary

    return ClosedLoopEpisodeSummary(**value)


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
      --bg: #0c1014;
      --panel: #121920;
      --panel-2: #18212b;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --line: #273342;
      --accent: #2dd4bf;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
      background: linear-gradient(180deg, #0c1014, #111827 45%, #0c1014);
      color: var(--text);
    }
    main { max-width: 1400px; margin: 0 auto; padding: 28px; }
    h1, h2, h3 { margin: 0 0 12px 0; }
    h2 { margin-top: 24px; }
    p { line-height: 1.5; }
    .muted, .small { color: var(--muted); }
    .small { font-size: 0.9rem; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 20px 0 28px;
    }
    .card, .episode {
      background: rgba(18, 25, 32, 0.9);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 12px 34px rgba(0, 0, 0, 0.28);
    }
    .label { color: var(--muted); font-size: 0.85rem; }
    .value { font-size: 1.7rem; margin-top: 6px; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: rgba(18, 25, 32, 0.78);
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--line);
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th { background: rgba(24, 33, 43, 0.94); }
    .gallery {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 14px;
    }
    img {
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      margin-top: 8px;
    }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    """


if __name__ == "__main__":
    raise SystemExit(main())
