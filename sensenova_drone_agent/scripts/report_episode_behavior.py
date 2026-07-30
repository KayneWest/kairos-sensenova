#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
import re
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES_ROOT = PROJECT_ROOT / "data" / "bc_sft" / "episodes"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "episode_behavior_report"
DEFAULT_PROGRESS_REPORT = PROJECT_ROOT / "output" / "bc_progress_report" / "index.html"
DEFAULT_EVAL_REPORT = PROJECT_ROOT / "output" / "closed_loop_eval_report" / "index.html"
DEFAULT_DASHBOARD_REPORT = PROJECT_ROOT / "output" / "training_dashboard" / "index.html"

RUN_RE = re.compile(r"^(overnight_bc_\d{8}T\d{6}Z)(?:_c\d{3}_e\d{3}_.+)?$")

ACTION_COLORS = {
    "hover": "#6c757d",
    "yaw_left": "#198754",
    "yaw_right": "#0d6efd",
    "ascend": "#20c997",
    "descend": "#fd7e14",
    "forward": "#2364aa",
    "backward": "#dc3545",
    "strafe_left": "#6f42c1",
    "strafe_right": "#d63384",
}


@dataclass
class EpisodeSummary:
    episode_id: str
    episode_dir: Path
    policy: str
    run_id: str
    actions: list[str]
    step_count: int
    status: str
    reason_counts: Counter[str]
    decision_family_counts: Counter[str]
    target_family_counts: Counter[str]
    decision_rich_steps: int
    mean_branch_score: float | None
    mean_goal_heading_deg: float | None
    mean_goal_right_m: float | None
    mean_front_clearance_m: float | None
    first_before: Path | None
    last_after: Path | None
    first_step_json: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", default=str(DEFAULT_EPISODES_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-episodes", type=int, default=10)
    parser.add_argument("--recent-runs", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    episodes_root = Path(args.episodes_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = collect_episode_summaries(episodes_root)
    report_path = out_dir / "index.html"
    report_path.write_text(
        build_report_html(
            out_dir=out_dir,
            episodes=episodes,
            sample_episodes=max(args.sample_episodes, 0),
            recent_runs=max(args.recent_runs, 0),
        ),
        encoding="utf-8",
    )
    print(json.dumps({"report_path": str(report_path)}, indent=2))
    return 0


def collect_episode_summaries(episodes_root: Path) -> list[EpisodeSummary]:
    if not episodes_root.is_dir():
        return []

    summaries: list[EpisodeSummary] = []
    for episode_dir in sorted(
        (path for path in episodes_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        episode_json = _load_json(episode_dir / "episode.json")
        episode_id = str(episode_json.get("episode_id", episode_dir.name))
        policy = str(episode_json.get("policy", "scripted"))
        status = str(episode_json.get("status", "unknown"))
        actions = [str(action) for action in episode_json.get("actions", [])]
        step_paths = sorted(episode_dir.rglob("step.json"))

        if not actions:
            actions = []
            for step_path in step_paths:
                obj = _load_json(step_path)
                if obj:
                    actions.append(str(obj.get("action", "unknown")))

        reason_counts: Counter[str] = Counter()
        decision_family_counts: Counter[str] = Counter()
        target_family_counts: Counter[str] = Counter()
        decision_rich_steps = 0
        branch_scores: list[float] = []
        heading_values: list[float] = []
        right_values: list[float] = []
        front_values: list[float] = []
        for step_path in step_paths:
            obj = _load_json(step_path)
            teacher = dict(obj.get("metadata", {}).get("teacher", {}))
            reason = teacher.get("reason")
            if reason:
                reason_counts[str(reason)] += 1
            decision_profile = dict(teacher.get("decision_profile", {}))
            family = decision_profile.get("family")
            if family:
                decision_family_counts[str(family)] += 1
            target_family = decision_profile.get("target_family")
            if target_family:
                target_family_counts[str(target_family)] += 1
            if bool(decision_profile.get("decision_rich", False)):
                decision_rich_steps += 1
            if decision_profile.get("branch_score") is not None:
                branch_scores.append(float(decision_profile["branch_score"]))
            goal_features = dict(teacher.get("goal_features", {}))
            if "heading_error_deg" in goal_features:
                heading_values.append(abs(float(goal_features["heading_error_deg"])))
            if "right_m" in goal_features:
                right_values.append(float(goal_features["right_m"]))
            depth = dict(teacher.get("depth_clearance_m", {}))
            if depth.get("front_m") is not None:
                front_values.append(float(depth["front_m"]))

        before_frames = sorted(episode_dir.rglob("frame_before.png"))
        after_frames = sorted(episode_dir.rglob("frame_after.png"))
        summaries.append(
            EpisodeSummary(
                episode_id=episode_id,
                episode_dir=episode_dir,
                policy=policy,
                run_id=_derive_run_id(episode_id),
                actions=actions,
                step_count=int(episode_json.get("step_count", len(step_paths) or len(actions))),
                status=status,
                reason_counts=reason_counts,
                decision_family_counts=decision_family_counts,
                target_family_counts=target_family_counts,
                decision_rich_steps=decision_rich_steps,
                mean_branch_score=mean(branch_scores) if branch_scores else None,
                mean_goal_heading_deg=mean(heading_values) if heading_values else None,
                mean_goal_right_m=mean(right_values) if right_values else None,
                mean_front_clearance_m=mean(front_values) if front_values else None,
                first_before=before_frames[0] if before_frames else None,
                last_after=after_frames[-1] if after_frames else None,
                first_step_json=step_paths[0] if step_paths else None,
            )
        )
    return summaries


def build_report_html(
    *,
    out_dir: Path,
    episodes: list[EpisodeSummary],
    sample_episodes: int,
    recent_runs: int,
) -> str:
    policy_counts = Counter(episode.policy for episode in episodes)
    action_counts = Counter(action for episode in episodes for action in episode.actions)
    reason_counts = Counter(reason for episode in episodes for reason, count in episode.reason_counts.items() for _ in range(count))
    decision_family_counts = Counter(
        family for episode in episodes for family, count in episode.decision_family_counts.items() for _ in range(count)
    )
    total_steps = sum(episode.step_count for episode in episodes)
    total_decision_rich = sum(episode.decision_rich_steps for episode in episodes)

    run_groups = group_runs(episodes)
    recent_run_ids = list(sorted(run_groups.keys(), reverse=True))[:recent_runs]
    run_rows = [run_groups[run_id] for run_id in recent_run_ids]
    samples = episodes[:sample_episodes]

    page = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'/>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        "<title>Episode Behavior Report</title>",
        "<style>",
        _style_block(),
        "</style>",
        "</head>",
        "<body><main>",
        "<h1>Episode Behavior Report</h1>",
        "<p class='muted'>Compares actual collected episode behavior across scripted and reactive-teacher runs.</p>",
        "<section>",
        "<h2>Navigation</h2>",
        _navigation_links(out_dir),
        "</section>",
        "<section class='cards'>",
        _metric_card("Episodes", str(len(episodes))),
        _metric_card("Steps", str(total_steps)),
        _metric_card("Policies", ", ".join(f"{k}:{v}" for k, v in sorted(policy_counts.items())) or "n/a"),
        _metric_card("Top Action", _top_label(action_counts)),
        _metric_card("Top Teacher Reason", _top_label(reason_counts)),
        _metric_card("Top Decision Family", _top_label(decision_family_counts)),
        _metric_card("Decision-Rich Steps", str(total_decision_rich)),
        "</section>",
        "<section>",
        "<h2>Policy Mix</h2>",
        _counter_table(policy_counts, "Policy", "Episodes"),
        "</section>",
        "<section>",
        "<h2>Action Mix</h2>",
        _action_bar_table(action_counts),
        "</section>",
        "<section>",
        "<h2>Teacher Reason Mix</h2>",
        _counter_table(reason_counts, "Reason", "Steps"),
        "</section>",
        "<section>",
        "<h2>Decision Family Mix</h2>",
        _counter_table(decision_family_counts, "Family", "Steps"),
        "</section>",
        "<section>",
        "<h2>Run Comparison</h2>",
        _run_table(run_rows),
        "</section>",
        "<section>",
        "<h2>Recent Episodes</h2>",
        _episode_cards(out_dir, samples),
        "</section>",
        "</main></body></html>",
    ]
    return "\n".join(page)


def group_runs(episodes: list[EpisodeSummary]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "episodes": 0,
            "steps": 0,
            "policies": Counter(),
            "actions": Counter(),
            "reasons": Counter(),
            "decision_families": Counter(),
            "decision_rich_steps": 0,
            "mean_branch_values": [],
            "mean_heading_values": [],
            "mean_front_values": [],
        }
    )
    for episode in episodes:
        row = groups[episode.run_id]
        row["run_id"] = episode.run_id
        row["episodes"] += 1
        row["steps"] += episode.step_count
        row["policies"][episode.policy] += 1
        row["actions"].update(episode.actions)
        row["reasons"].update(episode.reason_counts)
        row["decision_families"].update(episode.decision_family_counts)
        row["decision_rich_steps"] += episode.decision_rich_steps
        if episode.mean_branch_score is not None:
            row["mean_branch_values"].append(episode.mean_branch_score)
        if episode.mean_goal_heading_deg is not None:
            row["mean_heading_values"].append(episode.mean_goal_heading_deg)
        if episode.mean_front_clearance_m is not None:
            row["mean_front_values"].append(episode.mean_front_clearance_m)

    for row in groups.values():
        row["mean_branch_score"] = mean(row["mean_branch_values"]) if row["mean_branch_values"] else None
        row["mean_heading_deg"] = mean(row["mean_heading_values"]) if row["mean_heading_values"] else None
        row["mean_front_clearance_m"] = mean(row["mean_front_values"]) if row["mean_front_values"] else None
    return groups


def _run_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>No run data available.</p>"
    parts = [
        "<table><thead><tr><th>Run</th><th>Policies</th><th>Episodes</th><th>Steps</th><th>Top Actions</th><th>Top Reasons</th><th>Decision-Rich</th><th>Mean Branch</th><th>Mean |Heading|</th><th>Mean Front Clearance</th></tr></thead><tbody>"
    ]
    for row in rows:
        parts.append(
            "<tr>"
            f"<td>{html.escape(str(row['run_id']))}</td>"
            f"<td>{html.escape(_counter_inline(row['policies']))}</td>"
            f"<td>{int(row['episodes'])}</td>"
            f"<td>{int(row['steps'])}</td>"
            f"<td>{html.escape(_counter_inline(row['actions'], max_items=3))}</td>"
            f"<td>{html.escape(_counter_inline(row['reasons'], max_items=3))}</td>"
            f"<td>{int(row['decision_rich_steps'])}</td>"
            f"<td>{html.escape(_format_float(row.get('mean_branch_score')))}</td>"
            f"<td>{html.escape(_format_float(row.get('mean_heading_deg')))} deg</td>"
            f"<td>{html.escape(_format_float(row.get('mean_front_clearance_m')))} m</td>"
            "</tr>"
        )
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _episode_cards(out_dir: Path, episodes: list[EpisodeSummary]) -> str:
    if not episodes:
        return "<p class='muted'>No episode samples available.</p>"
    cards = ["<div class='gallery'>"]
    for episode in episodes:
        cards.append("<article class='episode'>")
        cards.append(f"<h3>{html.escape(episode.episode_id)}</h3>")
        cards.append(
            "<p>"
            f"<strong>Policy:</strong> {html.escape(episode.policy)}<br/>"
            f"<strong>Status:</strong> {html.escape(episode.status)}<br/>"
            f"<strong>Run:</strong> {html.escape(episode.run_id)}<br/>"
            f"<strong>Actions:</strong> {html.escape(', '.join(episode.actions))}<br/>"
            f"<strong>Reasons:</strong> {html.escape(_counter_inline(episode.reason_counts, max_items=4))}<br/>"
            f"<strong>Decision Families:</strong> {html.escape(_counter_inline(episode.decision_family_counts, max_items=4))}<br/>"
            f"<strong>Mean Branch Score:</strong> {html.escape(_format_float(episode.mean_branch_score))}"
            "</p>"
        )
        if episode.first_step_json is not None:
            cards.append(
                f"<p><a href='{html.escape(_relpath(out_dir, episode.first_step_json))}'>first step.json</a></p>"
            )
        if episode.first_before is not None:
            cards.append(
                f"<img src='{html.escape(_relpath(out_dir, episode.first_before))}' alt='first frame' loading='lazy'/>"
            )
        if episode.last_after is not None:
            cards.append(
                f"<img src='{html.escape(_relpath(out_dir, episode.last_after))}' alt='last frame' loading='lazy'/>"
            )
        cards.append("</article>")
    cards.append("</div>")
    return "\n".join(cards)


def _counter_table(counter: Counter[str], key_label: str, value_label: str) -> str:
    if not counter:
        return "<p class='muted'>No data.</p>"
    rows = [f"<table><thead><tr><th>{html.escape(key_label)}</th><th>{html.escape(value_label)}</th></tr></thead><tbody>"]
    for key, value in counter.most_common():
        rows.append(f"<tr><td>{html.escape(str(key))}</td><td>{int(value)}</td></tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _action_bar_table(counter: Counter[str]) -> str:
    if not counter:
        return "<p class='muted'>No action data.</p>"
    total = sum(counter.values())
    rows = ["<table><thead><tr><th>Action</th><th>Count</th><th>Share</th></tr></thead><tbody>"]
    for action, count in counter.most_common():
        share = 0.0 if total <= 0 else (count / total)
        color = ACTION_COLORS.get(action, "#777")
        rows.append(
            "<tr>"
            f"<td>{html.escape(action)}</td>"
            f"<td>{count}</td>"
            f"<td><div class='bar-wrap'><div class='bar' style='width:{100.0 * share:.1f}%; background:{color};'></div></div> {100.0 * share:.1f}%</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _metric_card(label: str, value: str) -> str:
    return (
        "<div class='card'>"
        f"<div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div>"
        "</div>"
    )


def _derive_run_id(episode_id: str) -> str:
    match = RUN_RE.match(episode_id)
    if match:
        return match.group(1)
    if episode_id.startswith("reactive_teacher_smoke"):
        return "reactive_teacher_smokes"
    if episode_id.startswith("episode_"):
        return "manual_scripted_bootstrap"
    return "manual_misc"


def _top_label(counter: Counter[str]) -> str:
    if not counter:
        return "n/a"
    key, value = counter.most_common(1)[0]
    return f"{key} ({value})"


def _counter_inline(counter: Counter[str], max_items: int = 5) -> str:
    if not counter:
        return "n/a"
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common(max_items))


def _navigation_links(out_dir: Path) -> str:
    links = ["<ul>"]
    for label, path in [
        ("Training Dashboard", DEFAULT_DASHBOARD_REPORT),
        ("BC Progress Report", DEFAULT_PROGRESS_REPORT),
        ("Closed-Loop Eval Report", DEFAULT_EVAL_REPORT),
    ]:
        if path.is_file():
            links.append(
                f"<li><a href='{html.escape(_relpath(out_dir, path))}'>{html.escape(label)}</a></li>"
            )
    links.append("</ul>")
    return "\n".join(links)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _relpath(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve())


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _style_block() -> str:
    return """
body {
  font-family: ui-sans-serif, system-ui, sans-serif;
  margin: 0;
  background: #f6f6f2;
  color: #171717;
}
main {
  max-width: 1180px;
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
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
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
  font-size: 24px;
  font-weight: 700;
  margin-top: 6px;
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
  vertical-align: top;
}
.bar-wrap {
  display: inline-block;
  width: 200px;
  height: 12px;
  background: #ececec;
  border-radius: 999px;
  margin-right: 8px;
  vertical-align: middle;
  overflow: hidden;
}
.bar {
  height: 12px;
  border-radius: 999px;
}
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
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
