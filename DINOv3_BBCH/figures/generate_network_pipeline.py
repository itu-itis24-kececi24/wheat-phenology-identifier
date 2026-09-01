"""Generate a publication-quality diagram of the DINOv3 BBCH pipeline."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
PNG_PATH = ROOT / "dinov3_bbch_network_pipeline.png"
PDF_PATH = ROOT / "dinov3_bbch_network_pipeline.pdf"

INK = "#24323D"
MUTED = "#5C6973"
BLUE = "#4C90C0"
BLUE_LIGHT = "#EAF3F8"
TEAL = "#3FA88C"
TEAL_LIGHT = "#E9F6F1"
GOLD = "#E3AE3D"
GOLD_LIGHT = "#FFF5D9"
CORAL = "#D96B52"
CORAL_LIGHT = "#FBECE8"
PURPLE = "#8373B5"
PURPLE_LIGHT = "#F0ECF8"
GRAY = "#DDE3E7"
GRAY_LIGHT = "#F6F8F9"
WHITE = "#FFFFFF"


def rounded_box(ax, xy, width, height, face, edge, title, lines=(), radius=0.012,
                title_size=9.0, text_size=7.2, linewidth=1.25, zorder=3):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.72, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=INK, zorder=zorder + 1)
    if lines:
        ax.text(x + width / 2, y + height * 0.35, "\n".join(lines), ha="center",
                va="center", fontsize=text_size, color=MUTED, linespacing=1.3,
                zorder=zorder + 1)
    return patch


def arrow(ax, start, end, color=INK, linewidth=1.35, style="-|>", dashed=False,
          connectionstyle="arc3,rad=0", zorder=5):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=10,
        linewidth=linewidth, color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=connectionstyle, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def panel(ax, y, height, label, title, face):
    patch = FancyBboxPatch(
        (0.018, y), 0.964, height,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=face, edgecolor="#D7DEE3", linewidth=0.9, zorder=0,
    )
    ax.add_patch(patch)
    ax.text(0.032, y + height - 0.035, f"({label})", fontsize=12, fontweight="bold",
            color=INK, va="top")
    ax.text(0.065, y + height - 0.035, title, fontsize=11.5, fontweight="bold",
            color=INK, va="top")


def wheat_thumbnail(ax, x, y, width, height, seed, label=None):
    rng = np.random.default_rng(seed)
    h, w = 95, 120
    yy = np.linspace(0, 1, h)[:, None]
    base = np.zeros((h, w, 3), dtype=float)
    base[..., 0] = 0.19 + 0.20 * yy
    base[..., 1] = 0.39 + 0.25 * (1 - yy)
    base[..., 2] = 0.16 + 0.11 * yy
    base += rng.normal(0, 0.035, base.shape)
    base = np.clip(base, 0, 1)
    ax.imshow(base, extent=(x, x + width, y, y + height), zorder=2, aspect="auto")
    for _ in range(34):
        sx = x + rng.uniform(0.02, 0.98) * width
        sy = y + rng.uniform(-0.02, 0.30) * height
        ex = sx + rng.uniform(-0.09, 0.09) * width
        ey = y + rng.uniform(0.55, 1.02) * height
        color = rng.choice(["#D2C16C", "#ADC85D", "#E0CE7D", "#7DAA4F"])
        ax.plot([sx, ex], [sy, ey], color=color, linewidth=rng.uniform(0.45, 1.05),
                alpha=0.92, zorder=3)
    ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=INK,
                           linewidth=0.85, zorder=4))
    if label:
        ax.text(x + width / 2, y - 0.012, label, ha="center", va="top",
                fontsize=7.0, color=MUTED)


def tile_icon(ax, x, y, width, height, overlap=True):
    ax.add_patch(Rectangle((x, y), width, height, facecolor="#B8CD77",
                           edgecolor=INK, linewidth=0.8, zorder=2))
    ncol, nrow = 4, 3
    for col in range(1, ncol):
        xpos = x + col * width / ncol
        ax.plot([xpos, xpos], [y, y + height], color=WHITE, linewidth=0.65,
                alpha=0.9, zorder=3)
    for row in range(1, nrow):
        ypos = y + row * height / nrow
        ax.plot([x, x + width], [ypos, ypos], color=WHITE, linewidth=0.65,
                alpha=0.9, zorder=3)
    if overlap:
        ax.add_patch(Rectangle((x + width * 0.16, y + height * 0.18),
                               width * 0.42, height * 0.44, facecolor="none",
                               edgecolor=CORAL, linewidth=1.25, zorder=4))
        ax.add_patch(Rectangle((x + width * 0.30, y + height * 0.32),
                               width * 0.42, height * 0.44, facecolor="none",
                               edgecolor=CORAL, linewidth=1.25, zorder=4))


def dense_token_icon(ax, x, y, width, height):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.003,rounding_size=0.006",
        facecolor=TEAL_LIGHT, edgecolor=TEAL, linewidth=1.0, zorder=2,
    ))
    positions = [
        (0.30, 0.30), (0.70, 0.30), (0.30, 0.70), (0.70, 0.70),
    ]
    for px, py in positions:
        ax.add_patch(Rectangle(
            (x + width * (px - 0.10), y + height * (py - 0.10)),
            width * 0.20, height * 0.20, facecolor=TEAL, edgecolor="none", zorder=3,
        ))
    ax.add_patch(Circle((x + width * 0.50, y + height * 0.50),
                        min(width, height) * 0.11, facecolor=CORAL,
                        edgecolor=WHITE, linewidth=0.7, zorder=4))
    ax.text(x + width / 2, y - 0.008, "CLS + 2x2 dense grid",
            ha="center", va="top", fontsize=6.4, color=MUTED)


def transformer_stack(ax, x, y, width, height):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=PURPLE_LIGHT, edgecolor=PURPLE, linewidth=1.3, zorder=2,
    ))
    ax.text(x + width / 2, y + height * 0.88, "Temporal Transformer x2",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=INK)
    inner_x = x + width * 0.13
    inner_w = width * 0.74
    labels = ["LN + MHA (8 heads)", "Residual addition", "LN + GELU FFN"]
    colors = [BLUE_LIGHT, GRAY_LIGHT, GOLD_LIGHT]
    edges = [BLUE, MUTED, GOLD]
    starts = [0.58, 0.38, 0.18]
    for label, face, edge, frac in zip(labels, colors, edges, starts):
        ax.add_patch(FancyBboxPatch(
            (inner_x, y + height * frac), inner_w, height * 0.13,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            facecolor=face, edgecolor=edge, linewidth=0.75, zorder=3,
        ))
        ax.text(x + width / 2, y + height * (frac + 0.065), label,
                ha="center", va="center", fontsize=6.8, color=INK, zorder=4)


def draw_figure():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(14.2, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(0.5, 0.982, "Fine-tuned DINOv3 spatio-temporal wheat phenology pipeline",
            ha="center", va="top", fontsize=15, fontweight="bold", color=INK)

    panel(ax, 0.565, 0.365, "a", "Fold-specific visual backbone adaptation", "#F7FAFC")
    panel(ax, 0.055, 0.465, "b", "Causal spatio-temporal BBCH classification", "#FCFBF7")

    # Panel (a): fold-specific fine-tuning.
    wheat_thumbnail(ax, 0.052, 0.665, 0.095, 0.145, seed=3, label="Training-station image")
    arrow(ax, (0.152, 0.737), (0.188, 0.737), color=BLUE)
    rounded_box(ax, (0.192, 0.650), 0.105, 0.170, BLUE_LIGHT, BLUE,
                "Overlapping tiles", (),
                title_size=8.2, text_size=6.8)
    tile_icon(ax, 0.215, 0.685, 0.058, 0.052)
    ax.text(0.244, 0.666, "224 x 224; random subset", ha="center",
            fontsize=6.4, color=MUTED)

    arrow(ax, (0.302, 0.737), (0.335, 0.737), color=BLUE)
    rounded_box(ax, (0.340, 0.635), 0.155, 0.200, BLUE_LIGHT, BLUE,
                "DINOv3 ViT-B/16", (),
                title_size=9.0, text_size=7.0)
    for idx in range(6):
        face = CORAL_LIGHT if idx == 5 else WHITE
        edge = CORAL if idx == 5 else "#AAB7C0"
        ax.add_patch(Rectangle((0.368 + idx * 0.017, 0.681), 0.012, 0.060,
                               facecolor=face, edgecolor=edge, linewidth=0.65, zorder=4))
    ax.text(0.417, 0.666, "blocks 1-11 frozen; block 12 + final LN trainable",
            ha="center", fontsize=6.15, color=MUTED)

    arrow(ax, (0.501, 0.737), (0.530, 0.737), color=TEAL)
    dense_token_icon(ax, 0.536, 0.680, 0.095, 0.105)

    arrow(ax, (0.637, 0.737), (0.666, 0.737), color=TEAL)
    rounded_box(ax, (0.672, 0.650), 0.125, 0.170, TEAL_LIGHT, TEAL,
                "Hierarchical attention", ("descriptor attention", "tile attention"),
                title_size=8.2, text_size=6.8)

    arrow(ax, (0.803, 0.737), (0.832, 0.737), color=CORAL)
    rounded_box(ax, (0.838, 0.650), 0.105, 0.170, CORAL_LIGHT, CORAL,
                "Temporary head", ("LN + dropout", "BBCH classifier"),
                title_size=8.2, text_size=6.8)
    ax.text(0.892, 0.622, "validation macro-F1 selection", ha="center",
            fontsize=6.8, color=MUTED)

    arrow(ax, (0.418, 0.635), (0.418, 0.548), color=BLUE, dashed=True)
    ax.text(0.426, 0.585, "save adapted backbone\n(discard temporary head)",
            ha="left", va="center", fontsize=7.0, color=BLUE)

    # Panel (b): daily sequence.
    dates = [("t-20", 7), ("...", 11), ("t-1", 15), ("t", 19)]
    x_positions = [0.045, 0.109, 0.173, 0.237]
    for (label, seed), xpos in zip(dates, x_positions):
        if label == "...":
            ax.text(xpos + 0.027, 0.393, "...", ha="center", va="center",
                    fontsize=15, color=MUTED)
        else:
            wheat_thumbnail(ax, xpos, 0.342, 0.054, 0.092, seed=seed, label=label)
    ax.text(0.161, 0.305, "21 calendar days; unavailable images are masked",
            ha="center", fontsize=6.8, color=MUTED)

    arrow(ax, (0.294, 0.390), (0.326, 0.390), color=BLUE)
    rounded_box(ax, (0.331, 0.325), 0.120, 0.145, BLUE_LIGHT, BLUE,
                "Dense image encoder", ("tiling -> adapted DINOv3", "5 descriptors per tile"),
                title_size=8.2, text_size=6.6)
    ax.text(0.391, 0.306, "applied independently to each day",
            ha="center", fontsize=6.5, color=MUTED)

    arrow(ax, (0.456, 0.390), (0.487, 0.390), color=TEAL)
    rounded_box(ax, (0.492, 0.325), 0.115, 0.145, TEAL_LIGHT, TEAL,
                "Spatial attention", ("patch -> tile", "one visual vector/day"),
                title_size=8.2, text_size=6.6)

    # Daily metadata branch and gated fusion.
    rounded_box(ax, (0.333, 0.105), 0.118, 0.115, GOLD_LIGHT, GOLD,
                "Daily metadata", ("normalized days", "since sowing"),
                title_size=8.0, text_size=6.6)
    rounded_box(ax, (0.492, 0.105), 0.115, 0.115, GOLD_LIGHT, GOLD,
                "Metadata MLP", ("hidden dim. 32",),
                title_size=8.0, text_size=6.6)
    arrow(ax, (0.456, 0.162), (0.487, 0.162), color=GOLD)
    gate_center = (0.631, 0.300)
    ax.add_patch(Circle(gate_center, 0.017, facecolor=GOLD_LIGHT,
                        edgecolor=GOLD, linewidth=1.1, zorder=5))
    ax.text(*gate_center, "+", ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK, zorder=6)
    ax.text(0.631, 0.266, r"gate $\sigma(g_d)$", ha="center", fontsize=6.6, color=MUTED)
    arrow(ax, (0.607, 0.162), (0.631, 0.281), color=GOLD,
          connectionstyle="arc3,rad=-0.13")
    arrow(ax, (0.607, 0.390), (0.631, 0.319), color=TEAL,
          connectionstyle="arc3,rad=0.08")

    # Tokens and transformer.
    arrow(ax, (0.650, 0.300), (0.680, 0.300), color=PURPLE)
    ax.text(0.669, 0.333, "daily tokens", ha="center", fontsize=6.5, color=MUTED)
    for idx, color in enumerate([CORAL, BLUE, BLUE, BLUE, BLUE]):
        ax.add_patch(Circle((0.695 + idx * 0.013, 0.300), 0.006,
                            facecolor=color, edgecolor=WHITE, linewidth=0.4, zorder=5))
    ax.text(0.721, 0.278, "CLS + position encoding", ha="center",
            fontsize=6.2, color=MUTED)
    transformer_stack(ax, 0.758, 0.238, 0.130, 0.230)
    arrow(ax, (0.750, 0.300), (0.755, 0.300), color=PURPLE)

    # Location branch and output.
    rounded_box(ax, (0.664, 0.105), 0.116, 0.095, GOLD_LIGHT, GOLD,
                "Station metadata", ("latitude, longitude", "elevation"),
                title_size=7.6, text_size=6.2)
    rounded_box(ax, (0.801, 0.105), 0.087, 0.095, GOLD_LIGHT, GOLD,
                "Location MLP", ("hidden dim. 16",),
                title_size=7.4, text_size=6.1)
    arrow(ax, (0.785, 0.152), (0.796, 0.152), color=GOLD)

    gate2 = (0.910, 0.255)
    ax.add_patch(Circle(gate2, 0.017, facecolor=GOLD_LIGHT,
                        edgecolor=GOLD, linewidth=1.1, zorder=5))
    ax.text(*gate2, "+", ha="center", va="center", fontsize=11,
            fontweight="bold", color=INK, zorder=6)
    ax.text(0.910, 0.286, r"$\sigma(g_l)$", ha="center",
            fontsize=6.2, color=MUTED)
    arrow(ax, (0.888, 0.152), (0.909, 0.236), color=GOLD,
          connectionstyle="arc3,rad=-0.10")
    arrow(ax, (0.888, 0.352), (0.910, 0.274), color=PURPLE,
          connectionstyle="arc3,rad=0.08")

    rounded_box(ax, (0.934, 0.206), 0.050, 0.150, CORAL_LIGHT, CORAL,
                "MLP", ("BBCH0", "BBCH1", "...", "BBCH8"),
                title_size=8.0, text_size=6.0)
    arrow(ax, (0.929, 0.255), (0.932, 0.255), color=CORAL)

    # Compact legend.
    ax.add_patch(Rectangle((0.053, 0.075), 0.012, 0.012, facecolor=CORAL_LIGHT,
                           edgecolor=CORAL, linewidth=0.8))
    ax.text(0.069, 0.081, "trainable during backbone adaptation", va="center",
            fontsize=6.5, color=MUTED)
    ax.add_patch(Rectangle((0.228, 0.075), 0.012, 0.012, facecolor=WHITE,
                           edgecolor="#AAB7C0", linewidth=0.8))
    ax.text(0.244, 0.081, "frozen", va="center", fontsize=6.5, color=MUTED)
    ax.plot([0.302, 0.328], [0.081, 0.081], color=BLUE, linewidth=1.2,
            linestyle=(0, (4, 3)))
    ax.text(0.333, 0.081, "saved fold-specific backbone", va="center",
            fontsize=6.5, color=MUTED)

    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    draw_figure()
