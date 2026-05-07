"""Bar plot: per-model accuracy on the MOSAIC eval set.

Bars are sorted by accuracy descending (no family grouping).

Saves both:
  - plots/model_accuracy.pdf  (vector — drop into the NeurIPS paper)
  - plots/model_accuracy.png  (high-DPI preview)
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt


# NeurIPS-style: serif body text, embeddable PDF fonts.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,   # editable text in the PDF
    "ps.fonttype": 42,
})


# (display_name, accuracy_pct, category)
# category ∈ {"baseline", "llm", "ours"}:
#   - baseline: existing specialized baseline (MolScribe)
#   - llm:      foundation-model image generators (single-shot)
#   - ours:     this paper's method
# Single-line labels — they're rotated 30° below so wrapping isn't needed.
ROWS = [
    ("MolScribe",                       12.0, "baseline"),
    ("GPT-image-2",                     16.6, "llm"),
    ("GPT-5 Image",                      9.7, "llm"),
    ("GPT-5 Image Mini",                 2.0, "llm"),
    ("Gemini 3 Pro Image Preview",      24.2, "llm"),
    ("Gemini 3.1 Flash Image Preview",  15.9, "llm"),
    ("Gemini 2.5 Flash Image",           2.5, "llm"),
    ("GPT-MAS-1",                       22.5, "ours"),
    ("GPT-MAS-4",                       33.3, "ours"),
    ("Gemini-MAS-1",                    42.6, "ours"),
    ("Gemini-MAS-4",                    53.9, "ours"),
    ("Gemini-MAS-7",                    56.9, "ours"),
]

# Three-category palette in the spirit of recent NeurIPS figures: a muted
# cool body (steel blue + light slate) sets the baseline field, with a
# tasteful teal accent reserved for our method. Avoids any saturated red,
# stays colorblind-distinguishable, and prints cleanly in B&W (the "ours"
# teal is darker than the slate baselines).
COLOR_FOR = {
    "baseline": "#A8B5C2",   # light cool slate (specialized baseline)
    "llm":      "#4C6E8A",   # refined steel blue (foundation models)
    "ours":     "#2A8B82",   # deep teal (ours)
}
LEGEND_LABEL = {
    "baseline": "Specialized baseline",
    "llm":      "Vanilla solver",
    "ours":     "Proposed (multi-agent system)",
}


def main() -> None:
    from matplotlib.patches import Patch

    # Ascending by accuracy — climbs to the headline "ours" bar on the right.
    rows = sorted(ROWS, key=lambda r: r[1])
    names      = [r[0] for r in rows]
    values     = [r[1] for r in rows]
    categories = [r[2] for r in rows]
    colors     = [COLOR_FOR[c] for c in categories]

    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    positions = list(range(len(rows)))
    bars = ax.bar(
        positions, values,
        color=colors, edgecolor="black", linewidth=0.6,
        width=0.72, zorder=3,
    )

    # Value labels on top of each bar.
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + max(values) * 0.018,
            f"{v:.1f}%",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    # Lift annotations in the SemiAnalysis / Artificial Analysis "growth-arrow"
    # style: a single smooth cubic Bezier that leaves the source bar going
    # straight up, arches over every in-between bar, and lands on the target
    # bar going straight down with an arrowhead. The lift label sits on the
    # arch apex.
    from matplotlib.path import Path
    from matplotlib.patches import FancyArrowPatch
    LIFT_ARROWS = [
        ("GPT-image-2",                "GPT-MAS-4"),
        ("Gemini 3 Pro Image Preview", "Gemini-MAS-7"),
    ]
    name_to_pos = {n: positions[i] for i, n in enumerate(names)}
    name_to_val = {n: values[i] for i, n in enumerate(names)}
    arrow_color = COLOR_FOR["ours"]
    vmax = max(values)
    for src, dst in LIFT_ARROWS:
        sx, sv = name_to_pos[src], name_to_val[src]
        dx, dv = name_to_pos[dst], name_to_val[dst]
        lift = dv - sv
        mult = dv / sv

        lo, hi = sorted((sx, dx))
        between_max = max(
            (v for p, v in zip(positions, values) if lo <= p <= hi),
            default=max(sv, dv),
        )
        # Endpoints sit a comfortable cushion above each bar's value label so
        # the arrow doesn't overprint any number. Targets the look of a
        # graceful "lift" annotation rather than a steep U-bracket.
        start_y = sv + vmax * 0.14
        end_y   = dv + vmax * 0.14
        # Cubic-Bezier control points pulled toward the chord midpoint
        # horizontally (not straight up) so the curve is a gentle arch instead
        # of a hard U. Vertical height clears every in-between value label.
        chord_dx = dx - sx
        ctrl_y = between_max + vmax * 0.16
        ctrl1 = (sx + chord_dx * 0.30, ctrl_y)
        ctrl2 = (dx - chord_dx * 0.30, ctrl_y)
        verts = [(sx, start_y), ctrl1, ctrl2, (dx, end_y)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        arrow = FancyArrowPatch(
            path=Path(verts, codes),
            arrowstyle="-|>,head_length=8,head_width=5",
            color=arrow_color,
            lw=1.8,
            zorder=5,
        )
        ax.add_patch(arrow)

        # Lift label on the arch apex (cubic Bezier with both controls at
        # ctrl_y peaks at y = 0.25*(start_y+end_y) + 0.75*ctrl_y).
        apex_y = 0.25 * (start_y + end_y) + 0.75 * ctrl_y
        midx = (sx + dx) / 2.0
        ax.text(
            midx, apex_y + vmax * 0.018,
            f"+{lift:.1f} pp  ({mult:.1f}×)",
            ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=arrow_color,
            zorder=6,
        )

    # Rotated tick labels — eliminates the wrapped-label overlap the previous
    # version had on long Gemini names.
    ax.set_xticks(positions)
    ax.set_xticklabels(
        names, rotation=30, ha="right", rotation_mode="anchor", fontsize=10,
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MOSAIC accuracy by image-generation model")
    # Extra headroom for the lift-arrow arches and their labels.
    ax.set_ylim(0, max(values) * 1.65)

    # Subtle horizontal gridlines, no chart-junk.
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Legend — only show categories that actually appear.
    seen = []
    for c in categories:
        if c not in seen:
            seen.append(c)
    legend_handles = [
        Patch(facecolor=COLOR_FOR[c], edgecolor="black", linewidth=0.6,
              label=LEGEND_LABEL[c])
        for c in seen
    ]
    # Legend on the left so it doesn't collide with the tall "ours" bar
    # at the right end of the ascending order.
    ax.legend(handles=legend_handles, loc="upper left", frameon=False,
              fontsize=9.5)

    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    pdf = os.path.join(out_dir, "model_accuracy.pdf")
    png = os.path.join(out_dir, "model_accuracy.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
