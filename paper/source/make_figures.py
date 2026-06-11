#!/usr/bin/env python3
"""
Regenerate all PRISM paper figures as vector PDFs (TrueType fonts).

Every value plotted here is copied verbatim from the versioned evidence
package (e1..e6). No internal PRISM mechanics are encoded; all series are
public timings, residuals, statuses, quantiles, and counts. Walk-forward and
Monte Carlo panels use recorded summary statistics only (no fabricated
time series or distributions).

Style: thin lines (<=1.5pt), direct labels, thin gridlines, navy / slate /
teal / electric-blue / violet palette with amber reserved for warning or
timeout status.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import pathlib

FIGS = pathlib.Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)

NAVY = "#172A54"
SLATE = "#64748B"
TEAL = "#0F766E"
BLUE = "#2563EB"
VIOLET = "#6D5BD0"
AMBER = "#B45309"
LINE = "#CBD5E1"
GRAYTXT = "#475569"
FILL = "#F1F5F9"

FS = 9
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS,
    "axes.titlesize": FS + 0.5,
    "axes.titleweight": "semibold",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 0.7,
    "axes.labelsize": FS - 0.5,
    "xtick.labelsize": FS - 1,
    "ytick.labelsize": FS - 1,
    "legend.fontsize": FS - 1.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.color": LINE,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.5,
    "lines.linewidth": 1.35,
    "lines.markersize": 4.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name):
    p = FIGS / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", p.name)


def dlabel(ax, x, y, text, color, dx=1.12, dy=1.0, ha="left", va="center", fs=None):
    ax.text(x * dx, y * dy, text, color=color, ha=ha, va=va,
            fontsize=(fs or FS - 1.5), fontweight="bold", family="sans-serif")


# ---------------------------------------------------------------
# Figure 2: multi-solver timing, two panels (e1 same-lane + e6 GPU lane)
# ---------------------------------------------------------------
def faster_cue(ax, direction="down", label="faster", x=0.022, y0=0.97, span=0.12):
    """Small axis-fraction arrow marking the better direction."""
    if direction == "down":
        ax.annotate("", xy=(x, y0 - span), xycoords="axes fraction",
                    xytext=(x, y0), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=GRAYTXT, lw=0.9))
        ax.text(x + 0.014, y0 - span / 2, label, transform=ax.transAxes,
                fontsize=FS - 1.8, color=GRAYTXT, family="sans-serif",
                ha="left", va="center")
    else:  # left
        ax.annotate("", xy=(x, y0), xycoords="axes fraction",
                    xytext=(x + span, y0), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=GRAYTXT, lw=0.9))
        ax.text(x + span + 0.012, y0, label, transform=ax.transAxes,
                fontsize=FS - 1.8, color=GRAYTXT, family="sans-serif",
                ha="left", va="center")


def fig_multisolver():
    """Single panel, inverted log time axis (faster on top). PRISM-CPU and
    PRISM-GPU are both first-class solid series. Per-N annotations report the
    better PRISM route against the fastest completed reference row (the PRISM
    stack routes between CPU and GPU configurations)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    series = [
        ("PRISM-CPU", NAVY, "o", 1.5,
         [(100, 1.6, 1), (500, 7.5, 1), (1000, 38.9, 1),
          (2000, 1421.0, 1), (5000, 1804.8, 1)]),
        ("PRISM-GPU", BLUE, "s", 1.5,
         [(100, 98.3, 1), (500, 94.3, 1), (1000, 99.4, 1),
          (2000, 112.2, 1), (5000, 130.2, 1)]),
        ("Clarabel", SLATE, "^", 1.2,
         [(100, 23.3, 1), (500, 355.6, 1), (1000, 1184.9, 1),
          (2000, 6347.0, 1), (5000, 74896.8, 0)]),
        ("OSQP", AMBER, "D", 1.2,
         [(100, 14.6, 1), (500, 329.1, 1), (1000, 60412.7, 0),
          (2000, 24908.9, 1), (5000, 101468.4, 0)]),
        ("SCS", VIOLET, "v", 1.2,
         [(100, 14.2, 1), (500, 180.8, 1), (1000, 860.6, 1),
          (2000, 6330.4, 1)]),
        ("MOSEK", TEAL, "p", 1.2,
         [(100, 10231.8, 1), (500, 3530.3, 1), (1000, 9448.5, 1),
          (2000, 28458.2, 1), (5000, 144794.6, 0)]),
    ]

    ax.axvspan(3800, 6600, color=LINE, alpha=0.3, zorder=0)

    for label, color, marker, lw, pts in series:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", color=color, lw=lw, zorder=2, alpha=0.95)
        done = [(x, y) for x, y, c in pts if c]
        fail = [(x, y) for x, y, c in pts if not c]
        if done:
            ax.plot(*zip(*done), marker, color=color, linestyle="None",
                    markerfacecolor=color, markeredgecolor=color,
                    markersize=4.6, zorder=3)
        if fail:
            ax.plot(*zip(*fail), marker, color=color, linestyle="None",
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.0, markersize=4.8, zorder=3)

    # routable-stack reference annotations: best PRISM route vs fastest
    # completed reference at each eligible N
    routes = [(100, 1.6, "8.9x vs SCS", 1.65, 0.62),
              (500, 7.5, "24.1x vs SCS", 1.16, 0.40),
              (1000, 38.9, "22.1x vs SCS", 1.16, 0.40),
              (2000, 1421.0, "4.5x vs SCS", 1.16, 0.40)]
    for n, y, lab, fx, fy in routes:
        ax.annotate(lab,
                    xy=(n, y), xytext=(n * fx, y * fy),
                    fontsize=FS - 2, color=GRAYTXT, fontweight="bold",
                    family="sans-serif", ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=GRAYTXT, lw=0.55,
                                    shrinkA=0, shrinkB=2))

    # direct labels
    dlabel(ax, 5000, 1804.8, "PRISM-CPU", NAVY, dy=1.9)
    ax.text(5000 * 1.12, 130.2, "PRISM-GPU\n(E6 reported-solve lane)",
            color=BLUE, fontsize=FS - 1.8, ha="left", va="center",
            family="sans-serif", fontweight="bold")
    dlabel(ax, 5000, 74896.8, "Clarabel", SLATE, dy=0.66)
    dlabel(ax, 5000, 101468.4, "OSQP", AMBER, dy=1.04)
    dlabel(ax, 1000, 860.6, "SCS", VIOLET, dx=0.58, dy=1.0)
    dlabel(ax, 5000, 144794.6, "MOSEK", TEAL, dy=1.55)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(80, 33000)
    ax.set_ylim(9e5, 0.45)   # inverted: faster sits on top
    ax.set_xlabel("Universe size $N$ (log scale)")
    ax.set_ylabel("Time (ms, log scale, inverted; lane per series)")
    ax.annotate("", xy=(0.025, 0.995), xycoords="axes fraction",
                xytext=(0.025, 0.875), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=GRAYTXT, lw=0.9))
    ax.text(0.038, 0.935, "faster", transform=ax.transAxes,
            fontsize=FS - 1.5, color=GRAYTXT, family="sans-serif",
            ha="left", va="center")
    ax.set_title("Multi-solver timing under the declared 60 s lane")
    ax.text(4950, 5.2e5, "deadline-feasibility row:\nno certified objective comparison",
            ha="center", va="bottom", fontsize=FS - 2.5, color=GRAYTXT,
            style="italic")

    handles = [
        Line2D([], [], marker="o", color=GRAYTXT, linestyle="None",
               markerfacecolor=GRAYTXT, label="completed"),
        Line2D([], [], marker="o", color=GRAYTXT, linestyle="None",
               markerfacecolor="white", markeredgewidth=1.0,
               label="timeout / late"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left",
              fontsize=FS - 2.2)
    save(fig, "fig_multisolver_timing.pdf")


# ---------------------------------------------------------------
# Figure 3: e6 GPU timing-lane decomposition
# ---------------------------------------------------------------
def fig_lanes():
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    Ns = [30, 100, 500, 1000, 2000, 5000]
    rep_solve = [101.6, 98.3, 94.3, 99.4, 112.2, 130.2]
    rep_wall = [104.4, 101.0, 96.9, 102.2, 115.4, 133.9]
    cold_wall = [1596.5, 134.5, 130.9, 110.1, 119.1, 153.4]

    ax.plot(Ns, rep_solve, "-s", color=BLUE, label="reported GPU solve interval")
    ax.plot(Ns, rep_wall, "-o", color=NAVY, label="repeated host wall-clock")
    ax.plot(Ns, cold_wall, "--^", color=AMBER, linewidth=1.2,
            label="cold host wall-clock")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(25, 110000)
    ax.set_ylim(2300, 55)   # inverted: faster sits on top
    ax.set_xlabel("Universe size $N$ (log scale)")
    ax.set_ylabel("Time (ms, log scale, inverted)")
    ax.set_title("PRISM-GPU timing-lane decomposition (E6)")
    # flat-floor band: the operational message of this lane
    ax.axhspan(94, 134, color=BLUE, alpha=0.07, zorder=0)
    ax.text(33, 88, "flat solve floor: 94\u2013130 ms while $N$ grows 167\u00d7",
            fontsize=FS - 2, color=BLUE, family="sans-serif",
            fontweight="bold", va="bottom")
    # host-overhead bracket at N=5,000 (reported vs repeated wall)
    ax.annotate("host overhead $\\leq$ 3.7 ms",
                xy=(5400, 132.0), xytext=(8200, 111.0),
                fontsize=FS - 2.2, color=GRAYTXT, family="sans-serif",
                va="center", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=GRAYTXT, lw=0.6,
                                shrinkA=2, shrinkB=1))
    # memory-class note
    ax.text(0.985, 0.04, "device memory: 5.6 kB \u2192 2.5 MB (Table 6)",
            transform=ax.transAxes, fontsize=FS - 2.2, color=GRAYTXT,
            family="sans-serif", ha="right", va="bottom", style="italic")
    ax.legend(loc="center right", frameon=True, fancybox=True,
              edgecolor=LINE, facecolor="white", framealpha=0.95,
              fontsize=FS - 1.6, borderpad=0.7)
    ax.annotate("", xy=(0.945, 0.985), xycoords="axes fraction",
                xytext=(0.945, 0.835), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=GRAYTXT, lw=0.9))
    ax.text(0.93, 0.91, "faster", transform=ax.transAxes,
            fontsize=FS - 1.5, color=GRAYTXT, family="sans-serif",
            ha="right", va="center")
    ax.annotate("first cold call carries one-time\nsetup cost at the public boundary",
                xy=(30, 1596.5), xytext=(110, 1300),
                fontsize=FS - 2, color=GRAYTXT, style="italic",
                arrowprops=dict(arrowstyle="-", color=GRAYTXT, lw=0.6))
    save(fig, "fig_timing_lanes.pdf")


# ---------------------------------------------------------------
# Figure 4: external diagnostics (e3)
# ---------------------------------------------------------------
def fig_residuals():
    fig, ax = plt.subplots(figsize=(6.7, 3.5))
    Ns = [100, 500, 1000, 5000, 25000, 100004]
    budget = [4.2e-9, 6.8e-9, 1.1e-8, 2.7e-8, 7.3e-8, 1.4e-7]
    bounds = [1.1e-9, 2.0e-9, 3.8e-9, 9.4e-9, 2.6e-8, 5.2e-8]
    station = [2.7e-7, 5.3e-7, 9.1e-7, 4.6e-6, 1.8e-5, 6.4e-5]
    compl = [8.4e-9, 1.2e-8, 2.4e-8, 6.8e-8, 2.1e-7, 8.9e-7]

    ax.plot(Ns, station, "-o", color=AMBER)
    ax.plot(Ns, compl, "-^", color=VIOLET)
    ax.plot(Ns, budget, "-s", color=NAVY)
    ax.plot(Ns, bounds, "-v", color=TEAL)

    dlabel(ax, 100004, 6.4e-5, "stationarity-style", AMBER, dx=1.1)
    dlabel(ax, 100004, 8.9e-7, "complementarity-style", VIOLET, dx=1.1, dy=0.68)
    dlabel(ax, 100004, 1.4e-7, "budget", NAVY, dx=1.1)
    dlabel(ax, 100004, 5.2e-8, "bounds", TEAL, dx=1.1, dy=0.78)

    ax.axhline(1e-4, color=GRAYTXT, lw=0.8, ls=(0, (6, 3)))
    ax.axhline(1e-6, color=GRAYTXT, lw=0.8, ls=(0, (1, 2)))
    ax.text(110, 1.35e-4, "budget gate $10^{-4}$", fontsize=FS - 2,
            color=GRAYTXT, ha="left", va="bottom")
    ax.text(110, 1.35e-6, "bound gate $10^{-6}$", fontsize=FS - 2,
            color=GRAYTXT, ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(80, 3.2e6)
    ax.set_ylim(5e-10, 6e-4)
    ax.set_xlabel("Universe size $N$ (log scale)")
    ax.set_ylabel("External residual (log scale)")
    ax.set_title("External post-solve diagnostics on returned portfolios")
    faster_cue(ax, "down", "tighter", x=0.022, y0=0.50)
    save(fig, "fig_quality_residuals.pdf")


# ---------------------------------------------------------------
# Figure 5: production queue hero (e5)
# ---------------------------------------------------------------
def fig_queue():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.9, 3.0))
    solvers = ["PRISM-GPU", "PRISM-CPU", "OSQP"]
    colors = [BLUE, NAVY, AMBER]
    completed = [500, 500, 4]
    p99 = [343.3, 392.3, 310000.0]

    bars = axa.bar(solvers, completed, color=colors, width=0.58, zorder=2)
    bars[2].set_hatch("///")
    bars[2].set_edgecolor(AMBER)
    bars[2].set_facecolor("white")
    bars[2].set_linewidth(0.8)
    for b, c in zip(bars, completed):
        axa.text(b.get_x() + b.get_width() / 2, c + 12, f"{c}/500",
                 ha="center", fontsize=FS - 1, color=GRAYTXT,
                 fontweight="bold", family="sans-serif")
    axa.set_ylim(0, 565)
    axa.set_ylabel("Accounts completed in window")
    axa.set_title("A. Queue completion", loc="left", fontsize=FS)
    axa.grid(axis="x", visible=False)

    bars = axb.bar(solvers, p99, color=colors, width=0.58, zorder=2)
    bars[2].set_hatch("///")
    bars[2].set_edgecolor(AMBER)
    bars[2].set_facecolor("white")
    bars[2].set_linewidth(0.8)
    labels = ["343 ms\nmiss 0.0%", "392 ms\nmiss 0.0%",
              "censored at cap\nmiss 99.2%"]
    for b, v, lab in zip(bars, p99, labels):
        axb.text(b.get_x() + b.get_width() / 2, v * 1.35, lab,
                 ha="center", fontsize=FS - 2, color=GRAYTXT,
                 family="sans-serif")
    axb.set_yscale("log")
    axb.set_ylim(1e2, 8e6)
    axb.set_ylabel("p99 solve time (ms, log scale)")
    axb.set_title("B. p99 timing and deadline misses", loc="left", fontsize=FS)
    axb.grid(axis="x", visible=False)
    faster_cue(axb, "down", "faster", x=0.05, y0=0.96)

    fig.suptitle("Production queue: 500 accounts, 10,000 instruments, declared 25-minute window",
                 fontsize=FS + 0.5, fontweight="bold", y=1.015)
    fig.text(0.115, -0.025,
             "Evidence E5. OSQP per-account values are censored at the recorded per-account cap; 4/500 accounts completed.",
             fontsize=FS - 2, color=GRAYTXT)
    fig.tight_layout()
    save(fig, "fig_queue_completion.pdf")


# ---------------------------------------------------------------
# Figure 6: E7 hard-scenario suite (real data)
# ---------------------------------------------------------------
def fig_hard_scenarios():
    import json
    e7 = json.load(open(pathlib.Path(__file__).parent /
                        "evidence/PRISM_EVIDENCE_E7_HARD_SCENARIOS_2026-06-11.json"))
    rows = e7["single_solves"]
    cert = e7["best_incumbent_objective"]
    s7 = e7["s7_scale_stress"]["results"]

    def get(scen, solver):
        for r in rows:
            if r["scenario"] == scen and r["solver"] == solver:
                return r

    def best_inc481(scen):
        cands = [r for r in rows if r["scenario"] == scen and
                 r["solver"] in ("OSQP", "Clarabel", "SCS") and r["feasible"]]
        return min(cands, key=lambda r: r["wall_ms"]) if cands else None

    def s7row(N, solver):
        for r in s7:
            if r["N"] == N and r["solver"] == solver:
                return r

    fig = plt.figure(figsize=(7.0, 9.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.25, 1.0], hspace=0.28)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    # ---------- Panel A: single solves, N=481 scenarios + scale rounds
    groups = []   # (label, gpu_row, cpu_row, inc_row, certobj)
    snames = {"S1": "S1 tax-aware\ntransition (N=481)",
              "S2": "S2 account\nrestrictions (N=481)",
              "S3": "S3 factor+turnover\ncontrol (N=481)"}
    for sc in ("S1", "S2", "S3"):
        groups.append((snames[sc], get(sc, "PRISM-GPU"), get(sc, "PRISM-CPU"),
                       best_inc481(sc), cert.get(sc)))
    for N in (9620, 24050, 48100):
        incs = [s7row(N, nm) for nm in ("Clarabel", "OSQP")]
        incs = [r for r in incs if r and r["feasible"]]
        inc = min(incs, key=lambda r: r["wall_ms"]) if incs else None
        co = min((r["objective"] for r in incs if r["objective"] is not None),
                 default=None)
        groups.append((f"scale stress\n(N={N:,})", s7row(N, "PRISM-GPU"),
                       s7row(N, "PRISM-CPU"), inc, co))
    for N in (96200, 192400, 384800):
        groups.append((f"routing probe\n(N={N:,})", s7row(N, "PRISM-GPU"),
                       s7row(N, "PRISM-CPU"), None, None))

    width = 0.31
    pitch = 1.34
    ys = [pitch * i for i in range(len(groups))][::-1]
    for gi, color, picker in [(0, BLUE, lambda g: g[1]),
                              (1, NAVY, lambda g: g[2])]:
        pos = [y + (gi - 1) * width for y in ys]
        vals, labs = [], []
        for g in groups:
            r = picker(g)
            vals.append(r["wall_ms"])
            v = r["wall_ms"]
            labs.append(f"{v/1000:.2f} s" if v >= 1000 else f"{v:.0f} ms")
        axa.barh(pos, vals, height=width * 0.88, color=color, zorder=2)
        for p, v, lab in zip(pos, vals, labs):
            axa.text(v * 1.38, p, lab, va="center", fontsize=FS - 2.2,
                     color=color, family="sans-serif", fontweight="bold")
    pos, vals, labs = [], [], []
    for y, g in zip(ys, groups):
        r = g[3]
        if r is None:
            axa.text(1.4, y + width, "incumbents not run (probe)",
                     va="center", fontsize=FS - 2.6, color=GRAYTXT,
                     style="italic")
            continue
        pos.append(y + width)
        v = r["wall_ms"]
        vals.append(v)
        labs.append((f"{v/1000:.1f} s" if v >= 1000 else f"{v:.0f} ms")
                    + f"  ({r['solver']})")
    axa.barh(pos, vals, height=width * 0.88, color=SLATE, zorder=2)
    for p, v, lab in zip(pos, vals, labs):
        axa.text(v * 1.38, p, lab, va="center", fontsize=FS - 2.2,
                 color=SLATE, family="sans-serif", fontweight="bold")
    axa.set_yticks(ys)
    axa.set_yticklabels([g[0] for g in groups], fontsize=FS - 1.8)
    axa.set_xscale("log")
    axa.set_xlim(1, 3.2e6)
    axa.set_xlabel("Solver-call wall-clock (ms, log)")
    axa.set_title("A. Constrained single solves: real data and real-calibrated scale stress",
                  loc="left", fontsize=FS)
    axa.grid(axis="y", visible=False)
    faster_cue(axa, "left", "faster", x=0.015, y0=0.965, span=0.085)
    axa.text(0.985, 0.965, "all PRISM gaps vs certified incumbent: +0.00%",
             transform=axa.transAxes, ha="right", va="center",
             fontsize=FS - 2, color=GRAYTXT, family="sans-serif",
             style="italic")
    leg = [Line2D([], [], color=c, lw=4, label=l)
           for l, c in [("PRISM-GPU", BLUE), ("PRISM-CPU", NAVY),
                        ("best incumbent", SLATE)]]
    axa.legend(handles=leg, frameon=False, loc="lower right",
               fontsize=FS - 2.2, handlelength=1.1)

    # ---------- Panel B: batches at two scales
    b1 = e7["s5_batch"]
    b2 = e7["s5b_batch_4810"]["results"]
    solvers = ["PRISM-GPU", "PRISM-CPU", "Clarabel", "OSQP"]
    colors = {"PRISM-GPU": BLUE, "PRISM-CPU": NAVY,
              "Clarabel": SLATE, "OSQP": "#9AA4B5"}
    gx = [0.0, 1.0]
    bw = 0.18
    for si, nm in enumerate(solvers):
        xs = [g + (si - 1.5) * bw for g in gx]
        vals = [b1[nm]["total_wall_s"], b2[nm]["total_wall_s"]]
        comp = [b1[nm]["completed"], b2[nm]["completed"]]
        axb.bar(xs, vals, width=bw * 0.9, color=colors[nm], zorder=2)
        for x, v, c in zip(xs, vals, comp):
            axb.text(x, v * 1.25, f"{v:.0f}s\n{c}", ha="center",
                     fontsize=FS - 2.6, color=GRAYTXT, family="sans-serif")
    axb.set_xticks(gx)
    axb.set_xticklabels(["200 accounts, N=481\n(+ SCS: 9s, 175/200)",
                         "50 accounts, N=4,810"], fontsize=FS - 1.5)
    axb.set_yscale("log")
    axb.set_ylim(0.5, 3e4)
    axb.set_ylabel("Queue wall-clock (s, log)")
    axb.set_title("B. Personalized-account batch throughput at two scales",
                  loc="left", fontsize=FS)
    axb.grid(axis="x", visible=False)
    faster_cue(axb, "down", "faster", x=0.04, y0=0.66)
    leg = [Line2D([], [], color=colors[nm], lw=4, label=nm) for nm in solvers]
    axb.legend(handles=leg, frameon=False, loc="upper left", ncol=2,
               fontsize=FS - 2.2, handlelength=1.1)

    fig.text(0.085, 0.04,
             f"Evidence E7: {e7['data']['n_assets']}-asset real-return universe "
             f"({e7['data']['period']}) and real-calibrated extensions; identical "
             "public objective recomputed externally on every returned weight vector.",
             fontsize=FS - 2, color=GRAYTXT)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.10)
    save(fig, "fig_hard_scenarios.pdf")


# ---------------------------------------------------------------
# Figure 7: claim-to-evidence matrix
# ---------------------------------------------------------------
def fig_claim_matrix():
    claims = [
        ("Completed-row speedup 4.5–24.1×", "timing", ["E1"]),
        ("N=5,000 deadline-feasibility row", "feasibility", ["E1"]),
        ("Objective gap 2.44% / 0.61% (LW)", "gap", ["E3"]),
        ("GPU timing-lane decomposition", "timing", ["E6"]),
        ("Residuals inside declared gates", "diagnostic", ["E3"]),
        ("Queue 500/500 in 109.5 s; 0 missed", "queue", ["E5"]),
        ("Constraint-burden stress curve", "feasibility", ["E5"]),
        ("Hard-scenario suite on real data", "timing", ["E7"]),
        ("Small-problem objective agreement", "agreement", ["E2"]),
        ("Audit-record provenance", "provenance", ["E5"]),
    ]
    arts = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
    cat_color = {"timing": NAVY, "feasibility": TEAL, "gap": VIOLET,
                 "queue": BLUE, "diagnostic": SLATE, "agreement": "#94A3B8",
                 "provenance": AMBER}

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    n = len(claims)
    glyph = {"timing": "T", "feasibility": "F", "gap": "G", "queue": "Q",
             "diagnostic": "D", "agreement": "A", "provenance": "P"}
    for i, (label, cat, used) in enumerate(claims):
        y = n - 1 - i
        ax.text(-0.62, y, label, ha="right", va="center", fontsize=FS - 1.5)
        for j, a in enumerate(arts):
            if a in used:
                ax.add_patch(Rectangle((j - 0.32, y - 0.32), 0.64, 0.64,
                                       facecolor=cat_color[cat],
                                       edgecolor="none", alpha=0.92))
                ax.text(j, y, glyph[cat], ha="center", va="center",
                        fontsize=FS - 2.2, color="white",
                        family="sans-serif", fontweight="bold")
            else:
                ax.add_patch(Rectangle((j - 0.32, y - 0.32), 0.64, 0.64,
                                       facecolor=FILL, edgecolor="none"))
    for j, a in enumerate(arts):
        ax.text(j, n - 0.25, a, ha="center", va="bottom",
                fontsize=FS - 1, family="monospace", color=GRAYTXT)
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none")
               for c in [NAVY, TEAL, VIOLET, BLUE, SLATE, "#94A3B8", AMBER]]
    labels = ["T timing", "F feasibility", "G objective gap", "Q queue",
              "D diagnostic", "A agreement", "P provenance"]
    ax.legend(handles, labels, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=FS - 2.2,
              handlelength=1.0, handleheight=1.0, columnspacing=1.2)
    ax.set_xlim(-6.9, 6.9)
    ax.set_ylim(-0.6, n + 0.4)
    ax.axis("off")
    ax.set_title("Claim-to-evidence matrix", fontsize=FS + 0.5,
                 fontweight="bold")
    save(fig, "fig_claim_matrix.pdf")


# ---------------------------------------------------------------
# Figure 8: CPU/GPU scale frontier (systems-boundary lane)
# ---------------------------------------------------------------
def fig_frontier():
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    Ns = [5000, 10000, 25000, 50000, 100004]
    cpu = [1804.8, 228.3, 1063.6, 2187.9, 2157.8]
    gpu = [140.7, 215.9, 458.3, 861.4, 2161.2]

    ax.plot(Ns, cpu, "-o", color=NAVY)
    ax.plot(Ns, gpu, "-s", color=BLUE)
    dlabel(ax, 5000, 1804.8, "PRISM-CPU", NAVY, dx=0.96, dy=1.35, ha="left")
    dlabel(ax, 5000, 140.7, "PRISM-GPU", BLUE, dx=1.25, dy=0.92, ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(110, 9000)
    ax.set_xlabel("Universe size $N$ (log scale)")
    ax.set_ylabel("Full-call time (ms, log scale)")
    ax.set_title("CPU/GPU scale frontier")
    ax.text(0.02, 0.95, "full-call systems lane", transform=ax.transAxes,
            fontsize=FS - 1.5, color=GRAYTXT, style="italic", va="top")
    ax.annotate("similar wall-clock values:\npublic-call systems effect,\nnot device equivalence",
                xy=(96000, 2350), xytext=(20000, 5300),
                ha="left", va="center",
                fontsize=FS - 2, color=GRAYTXT, style="italic",
                arrowprops=dict(arrowstyle="-", color=GRAYTXT, lw=0.6))
    save(fig, "fig_scale_frontier.pdf")


if __name__ == "__main__":
    fig_multisolver()
    fig_lanes()
    fig_residuals()
    fig_queue()
    fig_hard_scenarios()
    fig_claim_matrix()
    fig_frontier()
    print("all figures regenerated")
