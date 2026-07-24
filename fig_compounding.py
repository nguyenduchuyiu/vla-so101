"""Compounding-error figure from the real probe logs.

Two panels with the actual measurements from runs/overfit (ckpt-4000):
  (a) Per-frame error: on-manifold probe (model fed GT expert states every frame,
      probe_onmanifold.log) stays flat ~1.2 deg; closed-loop rollout (model fed its
      own output, probe_cl_s0.log) grows 0.5 -> 22 deg. Same model, same start state,
      only the input regime differs. That gap IS covariate shift.
  (b) shoulder_pan angle: GT expert holds at ~30 deg while the closed-loop rollout
      climbs to ~46 deg then wanders -- a concrete joint running away while the
      expert stays put.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CL_LOG = Path("outputs/probe_cl_s0.log")
OM_LOG = Path("outputs/probe_onmanifold.log")

_re_cl = re.compile(r"r(\d+) gf(\d+) chunkMAE=\s*([\d.]+) trackMAE=\s*([\d.]+) reached=\[([^\]]+)\] gt=\[([^\]]+)\]")
_re_om = re.compile(r"f(\d+) [A-Z_]+\s+armMAE=\s*([\d.]+)")


def parse_cl():
    frames, track, sh_reached, sh_gt = [], [], [], []
    for line in CL_LOG.read_text().splitlines():
        m = _re_cl.match(line)
        if not m:
            continue
        frames.append(int(m.group(2)))
        track.append(float(m.group(4)))
        sh_reached.append(float(m.group(5).split(",")[0]))
        sh_gt.append(float(m.group(6).split(",")[0]))
    return (np.array(frames), np.array(track),
            np.array(sh_reached), np.array(sh_gt))


def parse_om():
    frames, mae = [], []
    for line in OM_LOG.read_text().splitlines():
        m = _re_om.match(line)
        if not m:
            continue
        frames.append(int(m.group(1)))
        mae.append(float(m.group(2)))
    return np.array(frames), np.array(mae)


def main():
    cl_f, cl_track, cl_sh, cl_gt = parse_cl()
    om_f, om_mae = parse_om()
    # align closed-loop x to GT frame index (gf==frame since 2 env steps == 1 GT frame)
    diverge_f = 50  # gf050, where trackMAE first exceeds ~2 deg (r09)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # --- panel (a): error growth ---
    ax1.plot(om_f, om_mae, "o-", color="#3a9d4a", ms=4, lw=1.8,
             label="on-manifold (fed GT states)  [teacher forcing probe]")
    ax1.plot(cl_f, cl_track, "s-", color="#c0392b", ms=4, lw=1.8,
             label="closed-loop (fed own output)  [eval rollout]")
    ax1.axvline(diverge_f, color="#888", ls="--", lw=1)
    ax1.annotate("diverge\ngf050", (diverge_f, 2.2), fontsize=9.5, color="#555", ha="center")
    ax1.axvspan(175, 200, color="#e07b1a", alpha=0.12)
    ax1.text(187, 21.5, "GT grasp\ncloses", fontsize=8.5, color="#b5651a", ha="center")
    ax1.set_xlabel("GT frame index")
    ax1.set_ylabel("error (deg)")
    ax1.set_title("(a) Same model, same start — only the input regime differs")
    ax1.set_ylim(0, 24)
    ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax1.grid(alpha=0.3)
    ax1.text(0.98, 0.03, f"on-manifold mean={om_mae.mean():.2f}°  max={om_mae.max():.2f}°\n"
            f"closed-loop max={cl_track.max():.1f}°",
            transform=ax1.transAxes, ha="right", va="bottom", fontsize=9, color="#333",
            bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.3"))

    # --- panel (b): shoulder_pan divergence ---
    ax2.plot(cl_f, cl_gt, "o-", color="#3a9d4a", ms=4, lw=1.8, label="GT expert (holds)")
    ax2.plot(cl_f, cl_sh, "s-", color="#c0392b", ms=4, lw=1.8, label="closed-loop reached (drifts)")
    ax2.axvline(diverge_f, color="#888", ls="--", lw=1)
    ax2.annotate(f"GT freezes at 30°\nmodel climbs +15°", (diverge_f, 33), fontsize=9.5,
                 color="#555", ha="center")
    ax2.set_xlabel("GT frame index")
    ax2.set_ylabel("shoulder_pan (deg)")
    ax2.set_title("(b) Compounding drift in a single joint (shoulder_pan)")
    ax2.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    ax2.grid(alpha=0.3)

    fig.suptitle("Compounding error / covariate shift — runs/overfit ckpt-4000 (real probe data)",
                 fontsize=13.5, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path("outputs/figs/compounding_error.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved {out}")
    print(f"on-manifold: n={len(om_f)} mean={om_mae.mean():.3f} max={om_mae.max():.3f}")
    print(f"closed-loop: n={len(cl_f)} trackMAE max={cl_track.max():.2f} sh_pan reached peak={cl_sh.max():.1f} gt~{cl_gt[cl_f<=80].mean():.1f}")


if __name__ == "__main__":
    main()