#!/usr/bin/env python3
"""Render PAPER_DRAFT.md to a self-contained HTML page with embedded figures."""
import base64
import re
from pathlib import Path

import markdown

SDA = Path("/home/mkrzus/kairos-sensenova/sensenova_drone_agent")
FIGS = SDA / "output" / "paper_figures"
OUT = SDA / "paper" / "paper_draft_v1_2.html"

md_text = (SDA / "docs" / "PAPER_DRAFT.md").read_text()

FIGURES = {
    "@@FIG0@@": ("fig0_architecture.png", "Figure 0 — The think-then-act loop: perceive / think / act lanes, "
                 "the four fixes in place (rank hinge, unit-norm plans, reward-detach, per-step plan readout), "
                 "and the plan-free act-time heads."),
    "@@FIG1@@": ("fig1_training_dynamics.png", "Figure 1 — Training dynamics: scorer fidelity correlation and "
                 "the timing phase transition (arms D/E)."),
    "@@FIG3@@": ("fig3_two_seed_scorer.png", "Figure 3 — The robot-scorer transient across checkpoints in two "
                 "seeds: inversion tracks the timing transition, then recovers."),
    "@@FIG2@@": ("fig2_closed_loop.png", "Figure 2 — The whole paper in four panels: pre-DAgger seed-1 win, "
                 "seed-2 reversal, and post-DAgger consistency in both seeds (n = 1000 per panel)."),
    "@@FIG4@@": ("fig4_trace_grid.png", "Figure 4 — Thinking in frames: decoded imagined futures on a held-out "
                 "expert context. Selected ≈ true plan; random and zeroed plans imagine visibly worse futures."),
    "@@FIG5@@": ("fig5_dagger_cycles.png", "Figure 5 — Self-improvement is a repair, not a ladder: think-success "
                 "across DAgger data recipes, two seeds each. Cycle 1 passes the strict gate twice; every recipe "
                 "containing second-round self-data fails, and pure replacement re-inverts selection."),
    "@@FIG6@@": ("fig6_decomposition.png", "Figure 6 — Left: the judge/imagination exchange — think-success follows "
                 "the imagination in all 8 cells (2 seeds x 4 configurations) and is invariant to the judge. Right: "
                 "value-guided diffusion under the good judge — an action-blind generative prior gives guidance "
                 "nothing to steer."),
}

# Insert placeholders after the paragraph that first cites each figure.
ANCHORS = [
    ("heads (§5).", "@@FIG0@@"),
    ("checkpoint selection.", "@@FIG1@@\n\n@@FIG3@@"),
    ("a reliable behavioral advantage.", "@@FIG2@@"),
    ("self-improvement ladder.", "@@FIG5@@"),
    ("second-order by comparison.", "@@FIG6@@"),
    ("rest on the latent-space audit.", "@@FIG4@@"),
]
for anchor, placeholder in ANCHORS:
    assert anchor in md_text, f"anchor not found: {anchor}"
    md_text = md_text.replace(anchor, anchor + "\n\n" + placeholder, 1)

body = markdown.markdown(md_text, extensions=["tables"])

for placeholder, (fname, caption) in FIGURES.items():
    b64 = base64.b64encode((FIGS / fname).read_bytes()).decode()
    fig_html = (f'<figure class="paperfig"><img src="data:image/png;base64,{b64}" '
                f'alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    body = body.replace(f"<p>{placeholder}</p>", fig_html)

# Wrap tables for horizontal scroll.
body = body.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")

CSS = """
:root {
  --ground: #fcfcfa; --ink: #1c1e21; --muted: #5a5d63; --hairline: #e3e2dd;
  --accent: #2a78d6; --counter: #ce3f3f; --card: #ffffff; --code-bg: #f1f0ec;
}
@media (prefers-color-scheme: dark) {
  :root { --ground: #17191c; --ink: #e6e6e2; --muted: #9a9da3; --hairline: #2c2f34;
          --accent: #6aa3e8; --counter: #ee7a76; --card: #f5f5f2; --code-bg: #24262b; }
}
:root[data-theme="dark"] { --ground: #17191c; --ink: #e6e6e2; --muted: #9a9da3; --hairline: #2c2f34;
          --accent: #6aa3e8; --counter: #ee7a76; --card: #f5f5f2; --code-bg: #24262b; }
:root[data-theme="light"] { --ground: #fcfcfa; --ink: #1c1e21; --muted: #5a5d63; --hairline: #e3e2dd;
          --accent: #2a78d6; --counter: #ce3f3f; --card: #ffffff; --code-bg: #f1f0ec; }

html { background: var(--ground); }
body {
  font-family: Charter, "Bitstream Charter", "Source Serif Pro", Georgia, serif;
  color: var(--ink); background: var(--ground);
  margin: 0; padding: 3.5rem 1.25rem 6rem; line-height: 1.55; font-size: 1.05rem;
}
.sheet { max-width: 1080px; margin: 0 auto; }
.sheet > h1, .sheet > h2, .sheet > h3, .sheet > p, .sheet > ul, .sheet > ol, .sheet > em, .sheet > hr {
  max-width: 66ch; margin-left: auto; margin-right: auto;
}
.sheet > p:first-of-type { /* the draft-version line */
  font-family: system-ui, sans-serif; font-size: 0.82rem; color: var(--muted);
  letter-spacing: 0.02em;
}
h1 {
  font-size: 2.05rem; line-height: 1.2; text-wrap: balance; font-weight: 700;
  margin: 0 auto 0.75rem; letter-spacing: -0.01em;
}
h2 {
  font-size: 1.35rem; margin: 2.8rem auto 0.8rem; text-wrap: balance;
  border-top: 1px solid var(--hairline); padding-top: 1.6rem;
}
h3 { font-size: 1.08rem; margin: 2rem auto 0.6rem; }
p { margin: 0 auto 1rem; }
ul, ol { padding-left: 1.4rem; margin-bottom: 1rem; }
li { margin-bottom: 0.45rem; }
strong { font-weight: 700; }
a { color: var(--accent); }
hr { border: none; border-top: 1px solid var(--hairline); margin: 2.5rem auto; }
code {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.82em; background: var(--code-bg); padding: 0.1em 0.35em;
  border-radius: 3px; word-break: break-word;
}
.tablewrap {
  overflow-x: auto; margin: 1.4rem 0 1.8rem; border: 1px solid var(--hairline);
  border-radius: 4px;
}
table {
  border-collapse: collapse; width: 100%; font-family: system-ui, sans-serif;
  font-size: 0.78rem; font-variant-numeric: tabular-nums; line-height: 1.35;
}
th, td { padding: 0.45rem 0.65rem; text-align: left; border-bottom: 1px solid var(--hairline); }
th {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); font-weight: 600; white-space: nowrap;
}
tbody tr:last-child td { border-bottom: none; }
.paperfig { margin: 2rem auto; max-width: 1080px; }
.paperfig img {
  display: block; width: 100%; max-width: 100%; height: auto;
  background: var(--card); border: 1px solid var(--hairline); border-radius: 4px;
  padding: 0.6rem; box-sizing: border-box;
}
.paperfig figcaption {
  font-family: system-ui, sans-serif; font-size: 0.8rem; color: var(--muted);
  margin-top: 0.55rem; max-width: 76ch; margin-left: auto; margin-right: auto;
}
"""

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>From Scene Priors to Decision-Quality Imagination</title>
<style>{CSS}</style></head><body>
<div class="sheet">
{body}
</div>
</body></html>
"""
OUT.write_text(html)
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
