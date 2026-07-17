#!/usr/bin/env python3
"""Render PAPER_DRAFT.md to a shareable PDF via WeasyPrint (A4, embedded figures)."""
import base64
from pathlib import Path

import markdown
from weasyprint import HTML

SDA = Path("/home/mkrzus/kairos-sensenova/sensenova_drone_agent")
FIGS = SDA / "output" / "paper_figures"
OUT_PDF = SDA / "paper" / "paper_draft_v1_2.pdf"

md_text = (SDA / "docs" / "PAPER_DRAFT.md").read_text()

FIGURES = {
    "@@FIG0@@": ("fig0_architecture.png", "Figure 0 — The think-then-act loop: perceive / think / act lanes, "
                 "the four fixes in place, and the plan-free act-time heads."),
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

body = body.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")

CSS = """
@page {
  size: A4; margin: 22mm 19mm 24mm;
  @bottom-center { content: counter(page) " / " counter(pages);
                   font-family: Helvetica, sans-serif; font-size: 8pt; color: #7a7d82; }
  @top-right { content: "Draft v1.2 — July 2026";
               font-family: Helvetica, sans-serif; font-size: 7.5pt; color: #9a9da3; }
}
@page :first { @top-right { content: none; } }
body { font-family: Charter, Georgia, serif; color: #1c1e21; font-size: 10pt; line-height: 1.5; }
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 6pt; letter-spacing: -0.01em; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; border-top: 0.5pt solid #c9c8c3; padding-top: 10pt;
     page-break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 5pt; page-break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; }
ul, ol { margin: 0 0 7pt; padding-left: 16pt; }
li { margin-bottom: 3pt; text-align: justify; }
body > p:first-of-type { font-family: Helvetica, sans-serif; font-size: 8pt; color: #5a5d63;
                         text-align: left; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 7.5pt; background: #f1f0ec;
       padding: 0 2pt; }
.tablewrap { margin: 8pt 0 10pt; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; font-family: Helvetica, sans-serif;
        font-size: 6.6pt; line-height: 1.3; }
th, td { padding: 2.5pt 4pt; text-align: left; border-bottom: 0.5pt solid #d9d8d3; }
th { font-size: 6pt; text-transform: uppercase; letter-spacing: 0.05em; color: #5a5d63; }
.paperfig { margin: 10pt 0 12pt; page-break-inside: avoid; }
.paperfig img { width: 100%; border: 0.5pt solid #d9d8d3; padding: 3pt; }
.paperfig figcaption { font-family: Helvetica, sans-serif; font-size: 8pt; color: #5a5d63;
                       margin-top: 4pt; text-align: left; }
hr { border: none; border-top: 0.5pt solid #c9c8c3; margin: 14pt 0; }
"""

html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
HTML(string=html).write_pdf(OUT_PDF)
print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size/1e6:.1f} MB)")
