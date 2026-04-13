"""
A2_4 — uncertainty-aware cross-plan DVH plotting.

Reads the aggregated DVH NPZs produced by ``A2_3/run_uncertainty.py`` and
``A2_4/run_uncertainty.py`` and reproduces the multi-plan comparison plots
with ±1σ shaded bands in matching colours. If the A2_3 aggregated file is
missing the 1-field plan is silently skipped.

Outputs under ``A2_4/output/``:
    dvh_{plan}_all_structures_uncertainty.png     per-plan, all structures
    dvh_compare_{STRUCT}_uncertainty.png          cross-plan, one structure
    dvh_compare_key_structures_uncertainty.png    GTV + 3 OARs, all plans

Usage:
    python3 A2_4/analyse_dvh_multiplan_uncertainty.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)
import dvh_utils  # noqa: E402


OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

PLAN_NPZS = {
    "1-field": os.path.join(PROJECT_ROOT, "A2_3", "output",
                            "uncertainty", "aggregated.npz"),
    "2-beam":  os.path.join(OUTPUT_DIR, "uncertainty", "2beam", "aggregated.npz"),
    "3-beam":  os.path.join(OUTPUT_DIR, "uncertainty", "3beam", "aggregated.npz"),
    "4-beam":  os.path.join(OUTPUT_DIR, "uncertainty", "4beam", "aggregated.npz"),
}

# Per-plan colour for cross-plan comparisons (same palette as the base
# analyse_dvh_multiplan.py script the user already finds readable).
PLAN_COLORS = {
    "1-field": "#7f7f7f",  # grey
    "2-beam":  "#1f77b4",  # blue
    "3-beam":  "#2ca02c",  # green
    "4-beam":  "#d62728",  # red
}
PLAN_LINESTYLES = {
    "1-field": "--",
    "2-beam":  "-",
    "3-beam":  "-.",
    "4-beam":  ":",
}

# Per-structure colour / display name for per-plan plots.
STRUCTURE_COLORS = {
    "tumour": "red",
    "ptv":    "magenta",
    "lung_r": "orange",
    "lung_l": "olive",
    "heart":  "blue",
    "cord":   "green",
    "body":   "grey",
}
STRUCTURE_DISPLAY = {
    "tumour": "GTVp",
    "ptv":    "PTV",
    "lung_r": "Right Lung",
    "lung_l": "Left Lung",
    "heart":  "Heart",
    "cord":   "Spinal Cord",
    "body":   "Body",
}
LEGEND_ORDER = ("tumour", "ptv", "lung_r", "lung_l", "heart", "cord", "body")
COMPARE_STRUCTURES = ("tumour", "ptv", "heart", "cord", "lung_r", "lung_l", "body")


# ------------------------------------------------------------------
def load_all_plans():
    plans = {}
    for name, path in PLAN_NPZS.items():
        if not os.path.isfile(path):
            print(f"[skip] {name}: {path} not found")
            continue
        centres, agg = dvh_utils.load_aggregated_npz(path)
        plans[name] = {"centres": centres, "agg": agg}
        print(f"[ok]   {name}: {path}")
    return plans


# ------------------------------------------------------------------
# Plot 1 — per-plan, all structures with ±1σ bands
# ------------------------------------------------------------------
def plot_plan_all_structures(plan_name, plan_payload, output_path):
    centres = plan_payload["centres"]
    agg = plan_payload["agg"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    n_replicates = None
    for key in LEGEND_ORDER:
        if key not in agg:
            continue
        stats = agg[key]
        if n_replicates is None:
            n_replicates = int(stats.get("n", 0))
        dvh_utils.plot_dvh_with_band(
            ax, centres, stats["mean"], stats["std"],
            color=STRUCTURE_COLORS.get(key, "black"),
            label=STRUCTURE_DISPLAY.get(key, key),
        )

    ax.set_xlabel("Dose (% of plan max)", fontsize=12)
    ax.set_ylabel("Volume (%)", fontsize=12)
    title = f"Cumulative DVHs — {plan_name}"
    if n_replicates:
        title += f"\nMean ± 1σ over {n_replicates} independent seeds"
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ------------------------------------------------------------------
# Plot 2 — cross-plan DVH comparison for a single structure
# ------------------------------------------------------------------
def plot_structure_across_plans(structure_key, plans, output_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted = False
    n_replicates = None
    for plan_name, payload in plans.items():
        agg = payload["agg"]
        if structure_key not in agg:
            continue
        centres = payload["centres"]
        stats = agg[structure_key]
        if n_replicates is None:
            n_replicates = int(stats.get("n", 0))
        dvh_utils.plot_dvh_with_band(
            ax, centres, stats["mean"], stats["std"],
            color=PLAN_COLORS.get(plan_name, None),
            linestyle=PLAN_LINESTYLES.get(plan_name, "-"),
            label=plan_name,
        )
        plotted = True
    if not plotted:
        return

    ax.set_xlabel("Dose (% of plan max)", fontsize=12)
    ax.set_ylabel("Volume (%)", fontsize=12)
    title = f"Cumulative DVH Comparison — {STRUCTURE_DISPLAY.get(structure_key, structure_key)}"
    if n_replicates:
        title += f"\nMean ± 1σ over {n_replicates} independent seeds"
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="best", frameon=True, title="Plan")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ------------------------------------------------------------------
# Plot 3 — GTV + 3 OARs across all plans on one figure
# ------------------------------------------------------------------
def plot_key_structures(plans, output_path):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    struct_linestyles = {
        "tumour": "-",
        "heart":  "--",
        "cord":   "-.",
        "lung_r": ":",
        "lung_l": (0, (3, 1, 1, 1)),  # dash-dot-dot
    }
    plotted = False
    for plan_name, payload in plans.items():
        agg = payload["agg"]
        centres = payload["centres"]
        color = PLAN_COLORS.get(plan_name, None)
        for struct_key in ("tumour", "heart", "cord", "lung_r", "lung_l"):
            if struct_key not in agg:
                continue
            stats = agg[struct_key]
            dvh_utils.plot_dvh_with_band(
                ax, centres, stats["mean"], stats["std"],
                color=color,
                linestyle=struct_linestyles[struct_key],
                linewidth=1.5, alpha=0.18,
                label=f"{plan_name} — {STRUCTURE_DISPLAY[struct_key]}",
            )
            plotted = True
    if not plotted:
        return
    ax.set_xlabel("Dose (% of plan max)")
    ax.set_ylabel("Volume (%)")
    ax.set_title("Cumulative DVH Comparison — key structures across plans (mean ± 1σ)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ------------------------------------------------------------------
# Console summary table
# ------------------------------------------------------------------
def print_summary(plans):
    print()
    print("=" * 96)
    print("ENSEMBLE DVH SUMMARY (mean ± 1σ over replicates, dose in % of plan max)")
    print("=" * 96)
    hdr = (f'{"Plan":<9s} {"Structure":<13s} '
           f'{"D95":>14s} {"D50":>14s} {"D02":>14s} {"Dmax":>14s}')
    print(hdr)
    print("-" * len(hdr))
    for plan_name, payload in plans.items():
        agg = payload["agg"]
        for key in LEGEND_ORDER:
            if key not in agg:
                continue
            stats = agg[key]

            def fmt(dname):
                m = stats.get(f"{dname}_mean", float("nan"))
                s = stats.get(f"{dname}_std",  float("nan"))
                if not np.isfinite(m):
                    return "      —       "
                return f"{m:6.2f} ± {s:4.2f}"
            print(f'{plan_name:<9s} {STRUCTURE_DISPLAY.get(key, key):<13s} '
                  f'{fmt("D95"):>14s} {fmt("D50"):>14s} {fmt("D02"):>14s} '
                  f'{fmt("Dmax"):>14s}')
        print()


# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plans = load_all_plans()
    if not plans:
        print("No aggregated NPZ files found. Run run_uncertainty.py first.")
        return

    for plan_name, payload in plans.items():
        out_path = os.path.join(
            OUTPUT_DIR,
            f"dvh_{plan_name.replace('-', '_')}_all_structures_uncertainty.png")
        plot_plan_all_structures(plan_name, payload, out_path)

    for struct_key in COMPARE_STRUCTURES:
        display = STRUCTURE_DISPLAY.get(struct_key, struct_key).replace(" ", "")
        out_path = os.path.join(OUTPUT_DIR, f"dvh_compare_{display}_uncertainty.png")
        plot_structure_across_plans(struct_key, plans, out_path)

    out_key = os.path.join(OUTPUT_DIR, "dvh_compare_key_structures_uncertainty.png")
    plot_key_structures(plans, out_key)

    print_summary(plans)
    print("\nDone.")


if __name__ == "__main__":
    main()
