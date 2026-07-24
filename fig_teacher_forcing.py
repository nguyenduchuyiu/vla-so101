"""Teacher-forcing vs closed-loop schematic.

Two-panel cartoon contrasting the two input regimes:
  (a) Training (teacher forcing): the model is fed the ground-truth expert state at
      every step. Its own predicted action does NOT become the next input, so a
      prediction error at step t cannot contaminate step t+1. Errors stay local.
  (b) Evaluation (closed loop / autoregressive): the model's predicted action is
      executed in the env, producing the next state, which is fed back as the next
      input. A small error at step t moves the state off the expert manifold, and
      that off-manifold state is what the model sees at t+1 -> errors compound.

This is the mechanism behind the covariate-shift failure in runs/overfit.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_GREEN = "#3a9d4a"   # expert / GT
C_BLUE = "#2f6fbd"     # model
C_ORANGE = "#e07b1a"  # action
C_RED = "#c0392b"      # error / off-manifold
C_GREY = "#888888"


def box(ax, x, y, w, h, text, face, edge=None, text_color="white", fs=13, lw=1.5):
    edge = edge or face
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor=face, edgecolor=edge, linewidth=lw))
    ax.text(x, y, text, ha="center", va="center", color=text_color, fontsize=fs, weight="bold")


def arrow(ax, x0, y0, x1, y1, color="#444", lw=2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=lw, linestyle=ls))


def main():
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- Panel (a): teacher forcing ----
    ax = axes[0]
    ax.set_title("(a) Training — teacher forcing: next input = ground-truth state",
                 fontsize=13, weight="bold", loc="left", pad=8)
    xs = [0.16, 0.39, 0.62, 0.85]
    # teacher bar
    ax.add_patch(FancyBboxPatch((0.07, 0.86), 0.87, 0.05,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                facecolor=C_GREEN, edgecolor=C_GREEN, alpha=0.25))
    ax.text(0.50, 0.885, "expert dataset (teacher) supplies all $s_t^*$", ha="center", va="center",
            fontsize=11, style="italic", color=C_GREEN)
    for t, x in enumerate(xs):
        # GT state from teacher
        box(ax, x, 0.74, 0.10, 0.07, f"$s_{{{t}}}^*$", C_GREEN)
        arrow(ax, x, 0.86, x, 0.78, color=C_GREEN, lw=1.6)
        # model
        box(ax, x, 0.58, 0.085, 0.07, "$f_\\theta$", C_BLUE)
        arrow(ax, x, 0.705, x, 0.615, color="#444")
        # predicted action
        box(ax, x, 0.40, 0.085, 0.06, f"$\\hat a_{{{t}}}$", C_ORANGE)
        arrow(ax, x, 0.545, x, 0.43, color="#444")
        # loss vs GT action
        ax.text(x, 0.26, f"$L_t$\nvs $a_{{{t}}}^*$", ha="center", va="center",
                fontsize=10, color=C_RED)
        arrow(ax, x, 0.37, x, 0.30, color=C_RED, lw=1.2)
    # emphasize: NO feedback between timesteps
    ax.text(0.50, 0.10, "model output $\\hat a_t$ is NOT fed back as the next input  "
            r"$\Rightarrow$  errors stay local, do not accumulate",
            ha="center", va="center", fontsize=11.5, color=C_GREY, style="italic")

    # ---- Panel (b): closed loop ----
    ax = axes[1]
    ax.set_title("(b) Evaluation — closed loop: next input = model's own output through env",
                 fontsize=13, weight="bold", loc="left", pad=8)
    # layout: s0 -> f -> a0 -> env -> s1 -> f -> a1 -> env -> s2
    xs = [0.08, 0.16, 0.24, 0.32, 0.44, 0.52, 0.60, 0.68, 0.80]
    box(ax, xs[0], 0.55, 0.085, 0.07, "$s_0$", C_GREEN)
    box(ax, xs[1], 0.55, 0.06, 0.06, "$f_\\theta$", C_BLUE, fs=11)
    box(ax, xs[2], 0.55, 0.075, 0.06, "$\\hat a_0$", C_ORANGE)
    box(ax, xs[3], 0.55, 0.06, 0.06, "env", C_GREY, fs=11)
    box(ax, xs[4], 0.55, 0.085, 0.07, "$\\hat s_1$", C_ORANGE, edge=C_RED)
    box(ax, xs[5], 0.55, 0.06, 0.06, "$f_\\theta$", C_BLUE, fs=11)
    box(ax, xs[6], 0.55, 0.075, 0.06, "$\\hat a_1$", C_ORANGE)
    box(ax, xs[7], 0.55, 0.06, 0.06, "env", C_GREY, fs=11)
    box(ax, xs[8], 0.55, 0.085, 0.07, "$\\hat s_2$", C_ORANGE, edge=C_RED)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + 0.045, 0.55, b - 0.045, 0.55, color="#444", lw=1.8)
    # growing error clouds on the model-visited states
    for i, x in enumerate([xs[4], xs[8]]):
        ax.add_patch(plt.Circle((x, 0.55), 0.07 + 0.02 * i, fill=False,
                               edgecolor=C_RED, lw=1.6 + 1.2 * i, alpha=0.5 + 0.2 * i, linestyle="--"))
    # feedback callout
    ax.annotate("", (xs[4], 0.49), (xs[4], 0.18),
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2))
    ax.text(xs[4] + 0.02, 0.30, "off-manifold state\nfed back as next input",
            ha="left", va="center", fontsize=10.5, color=C_RED)
    ax.text(0.50, 0.10, r"small error at $t$ $\Rightarrow$ $\hat s_{t+1}$ leaves expert manifold "
            r"$\Rightarrow$ model extrapolates $\Rightarrow$ drift compounds",
            ha="center", va="center", fontsize=11.5, color=C_RED, style="italic")

    fig.suptitle("Why an overfit policy can still fail: teacher forcing hides compounding error",
                 fontsize=14.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path("outputs/figs/teacher_forcing.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()