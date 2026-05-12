"""Fig. 1 — Causal-VITGN architecture block diagram.

Shows the four-module pipeline and highlights the new Structural Causal
Layer (red) that turns this from a predictive model into one capable of
counterfactual reasoning.

Run:
    PYTHONPATH=. python scripts/figures/fig1_architecture.py
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from scripts.figures._utils import save, setup_matplotlib


def main():
    setup_matplotlib()

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    blocks = [
        ("Temporal\nGraph\nEncoder",  0.4, "#dbe9ff", "#1f4e96"),
        ("Variational\nLatent\nEncoder", 2.6, "#dbe9ff", "#1f4e96"),
        ("Structural\nCausal\nLayer", 4.8, "#ffd6d6", "#9b2222"),   # NEW
        ("Neural\nHawkes\nDecoder",   7.0, "#dbe9ff", "#1f4e96"),
    ]
    w, h = 1.8, 1.6
    y_bot = 0.7

    for text, x, face, edge in blocks:
        rect = mpatches.FancyBboxPatch(
            (x, y_bot),
            w, h,
            boxstyle="round,pad=0.06",
            facecolor=face,
            edgecolor=edge,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y_bot + h / 2, text,
                ha="center", va="center", fontsize=10, color=edge,
                fontweight="bold")

    arrows_y = y_bot + h / 2
    for x_start, x_end in [(2.2, 2.6), (4.4, 4.8), (6.6, 7.0)]:
        ax.annotate(
            "", xy=(x_end, arrows_y), xytext=(x_start, arrows_y),
            arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5),
        )

    # Inputs label (left)
    ax.text(0.05, arrows_y, "Cascade\nedges +\ntimes",
            ha="left", va="center", fontsize=9, color="#333")
    ax.annotate("", xy=(0.4, arrows_y), xytext=(0.05, arrows_y),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2))

    # Output label (right)
    ax.text(9.55, arrows_y, r"$\lambda(t)$" "\nintensity",
            ha="left", va="center", fontsize=9, color="#333")
    ax.annotate("", xy=(9.5, arrows_y), xytext=(8.8, arrows_y),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.2))

    # Title + caption-style annotation
    ax.set_title(
        "Causal-VITGN architecture · the SCM layer (red) supports "
        "counterfactual inference",
        pad=14,
    )
    ax.text(
        5.0, 0.15,
        r"Loss:  $\mathcal{L} = \mathcal{L}_{\mathrm{NLL}} "
        r"+ \beta\,\mathrm{KL}(q\|p) "
        r"+ \lambda_{\mathrm{DAG}}\,h(A)^2 "
        r"+ \lambda_{\mathrm{sp}}\,\|A\|_1$",
        ha="center", va="center", fontsize=10, color="#222",
        style="italic",
    )

    save(fig, "fig1_architecture")
    plt.close(fig)


if __name__ == "__main__":
    main()
