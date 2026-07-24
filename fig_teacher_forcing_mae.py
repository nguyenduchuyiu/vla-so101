"""Teacher-forcing ablation: per-frame arm MAE, on-manifold vs closed loop.

Plots the decisive ablation behind the runs/overfit closed-loop failure.
Two regimes, same model (ckpt-4000), same nominal episode (obj 0):

  - Teacher forcing / on-manifold (probe_onmanifold.log): at every GT frame we
    feed the exact expert state (image+proprio) and measure the predicted chunk
    MAE vs the GT future. Errors stay local and small -- the model is accurate
    ON the expert manifold.
  - Closed loop (probe_cl_s0.log): the model's own action is executed, the
    reached state becomes the next input, and we measure trackMAE = reached
    state vs GT state. Small early errors push the state off-manifold and the
    error compounds.

Gap between the two curves = compounding covariate shift, the thing teacher
forcing hides.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_GREEN = "#3a9d4a"   # teacher forcing / on-manifold
C_RED = "#c0392b"      # closed loop / compounding
C_BLUE = "#2f6fbd"

RE_ON = re.compile(r"f(\d+)\s+\S+\s+armMAE=\s*([\d.]+)")
RE_CL = re.compile(r"gf(\d+)\s+chunkMAE=\s*([\d.]+)\s+trackMAE=\s*([\d.]+)")


def parse_on(path):
    f, m = [], []
    for line in Path(path).read_text().splitlines():
        g = RE_ON.search(line)
        if g:
            f.append(int(g.group(1))); m.append(float(g.group(2)))
    return f, m


def parse_cl(path):
    f, track = [], []
    for line in Path(path).read_text().splitlines():
        g = RE_CL.search(line)
        if g:
            f.append(int(g.group(1))); track.append(float(g.group(3)))
    return f, track


def main():
    on_f, on_m = parse_on("outputs/probe_onmanifold.log")
    cl_f, cl_m = parse_cl("outputs/probe_cl_s0.log")

    fig, ax = plt.subplots(figsize=(10, 5.4))

    # phase background bands (REACH_PICK up to ~f175, GRASP f175-205, LIFT >205)
    phases = [("REACH_PICK", 0, 175, "#f4f4f4"),
              ("GRASP", 175, 205, "#fbeaea"),
              ("LIFT / PLACE", 205, 330, "#eef4fb")]
    for name, x0, x1, col in phases:
        ax.axvspan(x0, x1, color=col, alpha=0.9, zorder=0)
        ax.text((x0 + x1) / 2, 24.5, name,
                ha="center", va="top", fontsize=9.5, color="#777", style="italic")

    ax.plot(on_f, on_m, "-o", color=C_GREEN, lw=2.0, ms=3.5,
            label=f"teacher forcing (on-manifold)  mean={sum(on_m)/len(on_m):.2f}°")
    ax.plot(cl_f, cl_m, "-o", color=C_RED, lw=2.0, ms=3.5,
            label=f"closed loop (autoregressive)  final={cl_m[-1]:.1f}°")

    # first-divergence marker from closed-loop log (r12 gf060, trackMAE 4.19)
    ax.axvline(60, color=C_RED, ls=":", lw=1.2, alpha=0.6)
    ax.annotate("first divergence\ngf060 (trackMAE 4.2°)", xy=(60, 4.2),
                xytext=(78, 7.5), fontsize=9, color=C_RED,
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1))

    ax.set_xlabel("ground-truth frame index", fontsize=11)
    ax.set_ylabel("arm MAE  (deg)", fontsize=11)
    ax.set_title("Teacher forcing hides compounding error — same overfit model, two input regimes",
                 fontsize=12.5, weight="bold", loc="left")
    ax.set_xlim(-5, 330)
    ax.set_ylim(0, 25)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.25, zorder=0)

    fig.tight_layout()
    out = Path("outputs/figs/teacher_forcing_mae.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}  (on-manifold {len(on_f)} pts, closed-loop {len(cl_f)} pts)")


if __name__ == "__main__":
    main()