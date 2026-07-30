#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "overnight_bc"
DEFAULT_EPISODES_ROOT = PROJECT_ROOT / "data" / "bc_sft" / "episodes"
DEFAULT_MANIFEST_SUMMARY = PROJECT_ROOT / "data" / "bc_sft" / "manifests" / "bc_manifest_summary.json"
DEFAULT_BASELINE_SUMMARY = PROJECT_ROOT / "output" / "bc_policy_baseline" / "train_summary.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "bc_progress_report"
DEFAULT_BEHAVIOR_REPORT = PROJECT_ROOT / "output" / "episode_behavior_report" / "index.html"
DEFAULT_EVAL_REPORT = PROJECT_ROOT / "output" / "closed_loop_eval_report" / "index.html"
DEFAULT_DASHBOARD_REPORT = PROJECT_ROOT / "output" / "training_dashboard" / "index.html"


@dataclass
class CyclePoint:
    run_id: str
    cycle_index: int
    collector_policy: str
    episodes: int
    examples: int
    train_examples: int
    val_examples: int
    val_episode_count: int
    best_val_loss: float | None
    best_val_acc: float | None
    final_val_acc: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT))
    parser.add_argument("--manifest-summary", default=str(DEFAULT_MANIFEST_SUMMARY))
    parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-episodes", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_root = Path(args.log_root).expanduser().resolve()
    episodes_root = Path(args.episodes_root).expanduser().resolve()
    manifest_summary = _load_json_if_exists(Path(args.manifest_summary).expanduser())
    baseline_summary = _load_json_if_exists(Path(args.baseline_summary).expanduser())
    cycle_points = collect_cycle_points(log_root)
    sample_episodes = collect_sample_episodes(episodes_root, max_count=max(args.sample_episodes, 0))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "index.html"
    report_path.write_text(
        build_report_html(
            out_dir=out_dir,
            cycle_points=cycle_points,
            manifest_summary=manifest_summary,
            baseline_summary=baseline_summary,
            sample_episodes=sample_episodes,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"report_path": str(report_path)}, indent=2))
    return 0


def collect_cycle_points(log_root: Path) -> list[CyclePoint]:
    points: list[CyclePoint] = []
    if not log_root.is_dir():
        return points

    for run_dir in sorted(path for path in log_root.iterdir() if path.is_dir()):
        summary = _load_json_if_exists(run_dir / "summary.json")
        cycle_summaries = list(summary.get("cycles", [])) if isinstance(summary, dict) else []
        cycle_summary_map = {
            int(cycle.get("cycle_index", -1)): cycle
            for cycle in cycle_summaries
            if isinstance(cycle, dict)
        }

        for cycle_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("cycle_")):
            try:
                cycle_index = int(cycle_dir.name.split("_", maxsplit=1)[1])
            except Exception:
                continue
            export_summary = _load_json_if_exists(cycle_dir / "export.stdout.log")
            train_summary = _load_json_if_exists(cycle_dir / "train.stdout.log")
            cycle_meta = cycle_summary_map.get(cycle_index, {})

            history = list(train_summary.get("history", [])) if isinstance(train_summary, dict) else []
            val_accs = [
                float(epoch["val"]["accuracy"])
                for epoch in history
                if isinstance(epoch, dict) and epoch.get("val") is not None
            ]
            points.append(
                CyclePoint(
                    run_id=run_dir.name,
                    cycle_index=cycle_index,
                    collector_policy=str(cycle_meta.get("collector_policy", "scripted")),
                    episodes=int(export_summary.get("num_episodes", 0)) if isinstance(export_summary, dict) else 0,
                    examples=int(export_summary.get("num_examples", 0)) if isinstance(export_summary, dict) else 0,
                    train_examples=int(export_summary.get("train_examples", 0)) if isinstance(export_summary, dict) else 0,
                    val_examples=int(export_summary.get("val_examples", 0)) if isinstance(export_summary, dict) else 0,
                    val_episode_count=int(export_summary.get("val_episode_count", 0)) if isinstance(export_summary, dict) else 0,
                    best_val_loss=_maybe_float(train_summary.get("best_metric")) if isinstance(train_summary, dict) else None,
                    best_val_acc=max(val_accs) if val_accs else None,
                    final_val_acc=val_accs[-1] if val_accs else None,
                )
            )
    return points


def collect_sample_episodes(episodes_root: Path, *, max_count: int) -> list[dict[str, Any]]:
    if not episodes_root.is_dir() or max_count <= 0:
        return []

    episode_dirs = sorted(
        (path for path in episodes_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:max_count]

    samples: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        episode_json = _load_json_if_exists(episode_dir / "episode.json")
        first_before = next(iter(sorted(episode_dir.rglob("frame_before.png"))), None)
        first_after = next(iter(sorted(episode_dir.rglob("frame_after.png"))), None)
        samples.append(
            {
                "episode_dir": episode_dir,
                "episode_id": str(episode_json.get("episode_id", episode_dir.name)) if isinstance(episode_json, dict) else episode_dir.name,
                "policy": str(episode_json.get("policy", "unknown")) if isinstance(episode_json, dict) else "unknown",
                "actions": list(episode_json.get("actions", [])) if isinstance(episode_json, dict) else [],
                "status": str(episode_json.get("status", "unknown")) if isinstance(episode_json, dict) else "unknown",
                "step_count": int(episode_json.get("step_count", 0)) if isinstance(episode_json, dict) else 0,
                "frame_before": first_before,
                "frame_after": first_after,
            }
        )
    return samples


def build_report_html(
    *,
    out_dir: Path,
    cycle_points: list[CyclePoint],
    manifest_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    sample_episodes: list[dict[str, Any]],
) -> str:
    latest_manifest = manifest_summary or {}
    latest_baseline = baseline_summary or {}
    latest_val_acc = _latest_val_accuracy(latest_baseline)

    sections = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>BC Progress Report</title>",
        "<style>",
        _style_block(),
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        "<h1>BC Progress Report</h1>",
        "<p class='muted'>Static report generated from overnight logs, manifest summaries, and training outputs.</p>",
        "<section>",
        "<h2>Navigation</h2>",
        _navigation_links(out_dir),
        "</section>",
        "<section class='cards'>",
        _metric_card("Episodes", str(latest_manifest.get("num_episodes", 0))),
        _metric_card("Examples", str(latest_manifest.get("num_examples", 0))),
        _metric_card("Train / Val", f"{latest_manifest.get('train_examples', 0)} / {latest_manifest.get('val_examples', 0)}"),
        _metric_card("Latest Val Acc", _format_percent(latest_val_acc)),
        _metric_card("Best Val Loss", _format_float(latest_baseline.get("best_metric"))),
        _metric_card("Val Episodes", str(latest_manifest.get("val_episode_count", 0))),
        _metric_card("Decision-Rich", f"{latest_manifest.get('decision_rich_examples', 0)} ({_format_percent(latest_manifest.get('decision_rich_fraction'))})"),
        _metric_card("Mean Branch Score", _format_float(latest_manifest.get("mean_branch_score"))),
        "</section>",
        "<section>",
        "<h2>Cycle Trends</h2>",
        _line_chart("Examples by Cycle", cycle_points, lambda point: point.examples, "#2364aa"),
        _line_chart("Best Val Accuracy by Cycle", cycle_points, lambda point: point.best_val_acc, "#198754", percent=True),
        _cycle_table(cycle_points),
        "</section>",
        "<section>",
        "<h2>Action Distribution</h2>",
        _action_table(dict(latest_manifest.get("counts_by_action", {}))),
        "</section>",
        "<section>",
        "<h2>Teacher Decision Distribution</h2>",
        _action_table(dict(latest_manifest.get("counts_by_teacher_reason", {}))),
        _action_table(dict(latest_manifest.get("counts_by_decision_family", {}))),
        "</section>",
        "<section>",
        "<h2>Recent Episodes</h2>",
        _episode_gallery(out_dir, sample_episodes),
        "</section>",
        "<section>",
        "<h2>Raw Artifacts</h2>",
        _artifact_list(out_dir, latest_manifest, latest_baseline, cycle_points),
        "</section>",
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(sections)


def _line_chart(title: str, points: list[CyclePoint], value_fn, color: str, percent: bool = False) -> str:
    data: list[tuple[str, float]] = []
    for point in points:
        value = value_fn(point)
        if value is None:
            continue
        data.append((f"{point.run_id[-6:]}:{point.cycle_index}", float(value)))

    if not data:
        return f"<h3>{html.escape(title)}</h3><p class='muted'>No data.</p>"

    width = 720
    height = 220
    pad_x = 40
    pad_y = 20
    values = [value for _, value in data]
    min_value = min(values)
    max_value = max(values)
    if abs(max_value - min_value) < 1e-9:
        max_value = min_value + 1.0

    def x_at(index: int) -> float:
        if len(data) == 1:
            return pad_x
        return pad_x + (width - 2 * pad_x) * (index / (len(data) - 1))

    def y_at(value: float) -> float:
        scaled = (value - min_value) / (max_value - min_value)
        return height - pad_y - scaled * (height - 2 * pad_y)

    polyline = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, (_, v) in enumerate(data))
    dots = "\n".join(
        f"<circle cx='{x_at(i):.1f}' cy='{y_at(v):.1f}' r='3' fill='{color}' />"
        for i, (_, v) in enumerate(data)
    )
    labels = "\n".join(
        (
            f"<text x='{x_at(i):.1f}' y='{height - 4}' text-anchor='middle' class='axis'>{html.escape(label)}</text>"
        )
        for i, (label, _) in enumerate(data)
    )
    value_labels = "\n".join(
        (
            f"<text x='{x_at(i):.1f}' y='{y_at(v) - 8:.1f}' text-anchor='middle' class='value'>{html.escape(_format_percent(v) if percent else _format_float(v))}</text>"
        )
        for i, (_, v) in enumerate(data)
    )

    return (
        f"<h3>{html.escape(title)}</h3>"
        f"<svg viewBox='0 0 {width} {height}' class='chart'>"
        f"<line x1='{pad_x}' y1='{height - pad_y}' x2='{width - pad_x}' y2='{height - pad_y}' class='grid' />"
        f"<line x1='{pad_x}' y1='{pad_y}' x2='{pad_x}' y2='{height - pad_y}' class='grid' />"
        f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{polyline}' />"
        f"{dots}{labels}{value_labels}</svg>"
    )


def _cycle_table(points: list[CyclePoint]) -> str:
    rows = [
        "<table><thead><tr><th>Run</th><th>Cycle</th><th>Policy</th><th>Episodes</th><th>Examples</th><th>Best Val Acc</th><th>Best Val Loss</th></tr></thead><tbody>"
    ]
    for point in points:
        rows.append(
            "<tr>"
            f"<td>{html.escape(point.run_id)}</td>"
            f"<td>{point.cycle_index}</td>"
            f"<td>{html.escape(point.collector_policy)}</td>"
            f"<td>{point.episodes}</td>"
            f"<td>{point.examples}</td>"
            f"<td>{html.escape(_format_percent(point.best_val_acc))}</td>"
            f"<td>{html.escape(_format_float(point.best_val_loss))}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _action_table(counts: dict[str, Any]) -> str:
    if not counts:
        return "<p class='muted'>No action counts available.</p>"
    rows = ["<table><thead><tr><th>Action</th><th>Count</th></tr></thead><tbody>"]
    for action, count in counts.items():
        rows.append(f"<tr><td>{html.escape(str(action))}</td><td>{int(count)}</td></tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _episode_gallery(out_dir: Path, samples: list[dict[str, Any]]) -> str:
    if not samples:
        return "<p class='muted'>No episodes available.</p>"

    cards = ["<div class='gallery'>"]
    for sample in samples:
        before = sample.get("frame_before")
        after = sample.get("frame_after")
        before_rel = _relpath(out_dir, before) if before is not None else ""
        after_rel = _relpath(out_dir, after) if after is not None else ""
        actions = ", ".join(str(action) for action in sample.get("actions", []))
        cards.append("<article class='episode'>")
        cards.append(f"<h3>{html.escape(str(sample.get('episode_id', 'episode')))}</h3>")
        cards.append(f"<p><strong>Policy:</strong> {html.escape(str(sample.get('policy', 'unknown')))}<br/>")
        cards.append(f"<strong>Status:</strong> {html.escape(str(sample.get('status', 'unknown')))}<br/>")
        cards.append(f"<strong>Steps:</strong> {int(sample.get('step_count', 0))}<br/>")
        cards.append(f"<strong>Actions:</strong> {html.escape(actions)}</p>")
        if before_rel:
            cards.append(f"<img src='{html.escape(before_rel)}' alt='frame before' loading='lazy'/>")
        if after_rel:
            cards.append(f"<img src='{html.escape(after_rel)}' alt='frame after' loading='lazy'/>")
        cards.append("</article>")
    cards.append("</div>")
    return "\n".join(cards)


def _artifact_list(
    out_dir: Path,
    manifest_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    cycle_points: list[CyclePoint],
) -> str:
    items = ["<ul>"]
    if manifest_summary:
        items.append(
            f"<li><a href='{html.escape(_relpath(out_dir, Path(manifest_summary['manifest_path'])))}'>Current Manifest</a></li>"
        )
    baseline_summary_path = DEFAULT_BASELINE_SUMMARY.resolve()
    if baseline_summary:
        items.append(
            f"<li><a href='{html.escape(_relpath(out_dir, baseline_summary_path))}'>Latest Baseline Train Summary</a></li>"
        )
    if DEFAULT_BEHAVIOR_REPORT.is_file():
        items.append(
            f"<li><a href='{html.escape(_relpath(out_dir, DEFAULT_BEHAVIOR_REPORT))}'>Episode Behavior Report</a></li>"
        )
    if DEFAULT_EVAL_REPORT.is_file():
        items.append(
            f"<li><a href='{html.escape(_relpath(out_dir, DEFAULT_EVAL_REPORT))}'>Closed-Loop Eval Report</a></li>"
        )
    if DEFAULT_DASHBOARD_REPORT.is_file():
        items.append(
            f"<li><a href='{html.escape(_relpath(out_dir, DEFAULT_DASHBOARD_REPORT))}'>Training Dashboard</a></li>"
        )
    for point in cycle_points[-6:]:
        summary_path = DEFAULT_LOG_ROOT / point.run_id / f"cycle_{point.cycle_index:03d}" / "train.stdout.log"
        if summary_path.is_file():
            items.append(
                f"<li><a href='{html.escape(_relpath(out_dir, summary_path))}'>{html.escape(point.run_id)} cycle {point.cycle_index} train log</a></li>"
            )
    items.append("</ul>")
    return "\n".join(items)


def _navigation_links(out_dir: Path) -> str:
    links = ["<ul>"]
    for label, path in [
        ("Training Dashboard", DEFAULT_DASHBOARD_REPORT),
        ("Episode Behavior Report", DEFAULT_BEHAVIOR_REPORT),
        ("Closed-Loop Eval Report", DEFAULT_EVAL_REPORT),
    ]:
        if path.is_file():
            links.append(
                f"<li><a href='{html.escape(_relpath(out_dir, path))}'>{html.escape(label)}</a></li>"
            )
    links.append("</ul>")
    return "\n".join(links)


def _metric_card(label: str, value: str) -> str:
    return (
        "<div class='card'>"
        f"<div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div>"
        "</div>"
    )


def _latest_val_accuracy(summary: dict[str, Any]) -> float | None:
    history = list(summary.get("history", [])) if isinstance(summary, dict) else []
    if not history:
        return None
    val = history[-1].get("val")
    if not isinstance(val, dict):
        return None
    return _maybe_float(val.get("accuracy"))


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _format_float(value: Any) -> str:
    value = _maybe_float(value)
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_percent(value: Any) -> str:
    value = _maybe_float(value)
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _relpath(base_dir: Path, target: Path) -> str:
    return os.path.relpath(Path(target).resolve(), start=base_dir.resolve())


def _style_block() -> str:
    return """
body {
  font-family: ui-sans-serif, system-ui, sans-serif;
  margin: 0;
  background: #f6f6f2;
  color: #171717;
}
main {
  max-width: 1120px;
  margin: 0 auto;
  padding: 24px;
}
h1, h2, h3 {
  margin: 0 0 12px 0;
}
section {
  margin-top: 28px;
}
.muted {
  color: #666;
}
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
}
.card {
  background: white;
  border: 1px solid #ddd;
  border-radius: 10px;
  padding: 14px;
}
.label {
  font-size: 12px;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.value {
  font-size: 28px;
  font-weight: 700;
  margin-top: 6px;
}
.chart {
  width: 100%;
  height: auto;
  background: white;
  border: 1px solid #ddd;
  border-radius: 10px;
  margin-bottom: 18px;
}
.grid {
  stroke: #bbb;
  stroke-width: 1;
}
.axis {
  font-size: 10px;
  fill: #666;
}
.value {
  font-size: 28px;
}
svg .value {
  font-size: 10px;
  fill: #333;
}
table {
  width: 100%;
  border-collapse: collapse;
  background: white;
  border: 1px solid #ddd;
}
th, td {
  padding: 10px;
  border-bottom: 1px solid #eee;
  text-align: left;
}
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 16px;
}
.episode {
  background: white;
  border: 1px solid #ddd;
  border-radius: 10px;
  padding: 12px;
}
.episode img {
  width: 100%;
  display: block;
  border-radius: 8px;
  margin-top: 8px;
}
a {
  color: #2364aa;
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
}
"""


if __name__ == "__main__":
    raise SystemExit(main())
