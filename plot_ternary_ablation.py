#!/usr/bin/env python3
"""Build out/ternary_ablation.pdf from out/ternary_c_results.json.

Three panels: (a) adaptive-C cliff 1-F vs L on two datasets with e^{-beta L}
overlaid, (b) adaptive vs random C on the text task, (c) fitted beta vs delta.
Style matches fidelity_curves.pdf (log-x, shaded 95% CI over 20 seeds).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
RES = OUT / "ternary_c_results.json"

C_PURPLE = "#9467bd"
C_TEAL = "#17becf"
FIT_GRAY = "#444444"


def _fit_line(L, mean, lo=1.5e-3, hi=0.30):
    """log(1-F) = c - beta L on the decay window; returns (beta, c, Ls)."""
    from scipy import stats as spstats
    pts = [(x, np.log(m)) for x, m in zip(L, mean) if lo < m <= hi]
    if len(pts) < 2:
        return None
    xs, ys = zip(*pts)
    res = spstats.linregress(xs, ys)
    return -float(res.slope), float(res.intercept), np.array(xs)


def main():
    res = json.loads(RES.read_text())
    L = np.array(res["L_values"], dtype=float)
    dims, labels, fits = res["dims"], res["labels"], res["fits"]
    acs = res["delta_sweep_dataset"]
    text = res["text_task"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.2, 3.5))

    # ---- panel (a): adaptive C cliff on the two datasets, exp fit overlaid ----
    series = [(acs, C_PURPLE, "D"), (text, C_TEAL, "p")]
    for d, color, mk in series:
        m = np.array(res["adaptive"][d]["mean"])
        ci = np.array(res["adaptive"][d]["ci"])
        lab = f"adaptive C, {labels[d].split(' (')[0]} ($d{{=}}{dims[d]}$)"
        axA.plot(L, m, color=color, marker=mk, ms=6, lw=1.6, label=lab)
        axA.fill_between(L, np.maximum(m - ci, 0), m + ci, color=color,
                         alpha=0.18, lw=0)
        fit = _fit_line(L, m)
        if fit:
            beta, c, xs = fit
            xx = np.linspace(xs.min(), L.max(), 100)
            axA.plot(xx, np.exp(c - beta * xx), color=color, ls=":", lw=1.4)
            axA.text(0.96, 0.92 - 0.10 * series.index((d, color, mk)),
                     rf"$\beta={beta:.1e}$, $R^2={fits[d]['r2']:.2f}$",
                     transform=axA.transAxes, ha="right", va="top",
                     fontsize=10.5, color=color)
    axA.set_xscale("log")
    axA.set_xlabel(r"queries  $L$")
    axA.set_ylabel(r"extraction error  $1-F$")
    axA.set_title("(a) Adaptive C: the ternary cliff")
    axA.set_ylim(bottom=0)
    axA.grid(alpha=0.3)
    axA.legend(frameon=False, fontsize=9.5, loc="upper right",
               bbox_to_anchor=(1.0, 0.74))

    # ---- panel (b): adaptive vs random C on the text task ----
    ma = np.array(res["adaptive"][text]["mean"]); ca = np.array(res["adaptive"][text]["ci"])
    mr = np.array(res["random"][text]["mean"]); cr = np.array(res["random"][text]["ci"])
    axB.plot(L, ma, color=C_PURPLE, marker="D", ms=6, lw=1.7, label="adaptive C")
    axB.fill_between(L, np.maximum(ma - ca, 0), ma + ca, color=C_PURPLE, alpha=0.18, lw=0)
    axB.plot(L, mr, color=C_PURPLE, marker="v", ms=6, lw=1.7, ls="--",
             markerfacecolor="white", label="random C")
    axB.fill_between(L, np.maximum(mr - cr, 0), mr + cr, color=C_PURPLE, alpha=0.10, lw=0)
    axB.set_xscale("log")
    axB.set_xlabel(r"queries  $L$")
    axB.set_ylabel(r"extraction error  $1-F$")
    axB.set_title(rf"(b) Adaptivity unlocks the rate ($d{{=}}{dims[text]}$)")
    axB.set_ylim(bottom=0)
    axB.grid(alpha=0.3)
    axB.legend(frameon=False, fontsize=10.5, loc="upper right")

    # ---- panel (c): fitted beta vs delta ----
    sweep = res["delta_sweep"]
    keys = sorted(sweep, key=lambda k: sweep[k]["delta"])
    deltas = np.array([sweep[k]["delta"] for k in keys])
    betas = np.array([sweep[k]["beta"] for k in keys])
    axC.plot(deltas, betas, color=C_PURPLE, marker="D", ms=7, lw=1.7)
    axC.set_xscale("log")
    axC.set_xlabel(r"abstain scale  $\delta$")
    axC.set_ylabel(r"fitted rate  $\beta$")
    axC.set_title(rf"(c) Rate vs. $\delta$ ({labels[acs].split(' (')[0]})")
    axC.grid(alpha=0.3, which="both")
    axC.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "ternary_ablation.pdf")
    plt.close(fig)
    print("wrote", OUT / "ternary_ablation.pdf")


if __name__ == "__main__":
    main()
