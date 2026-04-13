"""
A2_4 — Monte Carlo uncertainty driver for multi-field photon plans.

Re-runs the optimised 2/3/4-beam plans produced by ``sweep_beams.py`` with
``N_REPLICATES`` independent RNG seeds, builds per-structure cumulative DVHs
directly from the TOPAS voxel-level dose CSVs, and aggregates replicates into
mean ± 1σ bands.

The optimised production files written by ``sweep_beams.py`` are used as
templates; histories, seed, OutputType and OutputFile are rewritten for each
replicate, everything else (beam geometry, weights, apertures) is preserved.

Templates expected:
    A2_4/output/optimised_2beam_optimised.txt
    A2_4/output/optimised_3beam_optimised.txt
    A2_4/output/optimised_4beam_optimised.txt

Outputs, per plan (under ``A2_4/output/uncertainty/{plan}/``):
    replicate_{kk}.txt   — TOPAS input for seed k
    replicate_{kk}.csv   — TOPAS voxel dose output
    replicate_{kk}.npz   — per-replicate cached DVHs
    aggregated.npz       — ensemble mean / 1σ / min / max (+ D-value stats)

Re-runs are cached: replicates whose ``.npz`` is already present are loaded
from disk. Pass ``--force`` to re-run every replicate. The single 1-field
plan is handled by ``A2_3/run_uncertainty.py`` so that A2_3 and A2_4 share
the same replicate store.

Usage:
    python3 A2_4/run_uncertainty.py
    python3 A2_4/run_uncertainty.py --force
"""

import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

sys.path.insert(0, SCRIPT_DIR)
import dvh_utils  # noqa: E402
from sweep_beams import load_ct_geometry, build_structure_masks  # noqa: E402


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
N_REPLICATES = 10
HISTORIES_PER_REPLICATE = 5_000_000
SEEDS = list(range(1, N_REPLICATES + 1))

PLANS = {
    "2beam": os.path.join(SCRIPT_DIR, "output", "optimised_2beam_optimised.txt"),
    "3beam": os.path.join(SCRIPT_DIR, "output", "optimised_3beam_optimised.txt"),
    "4beam": os.path.join(SCRIPT_DIR, "output", "optimised_4beam_optimised.txt"),
}

UNCERT_ROOT = os.path.join(SCRIPT_DIR, "output", "uncertainty")
CT_DIR      = os.path.join(PROJECT_ROOT, "CTData")

STRUCTURE_KEYS = ("tumour", "ptv", "lung_r", "lung_l", "heart", "cord", "body")

FORCE_FLAG   = "--force"
RESCORE_FLAG = "--rescore"

STRUCT_DISPLAY = {
    "tumour": "GTVp",
    "ptv":    "PTV",
    "lung_r": "Right Lung",
    "lung_l": "Left Lung",
    "heart":  "Heart",
    "cord":   "Spinal Cord",
    "body":   "Body",
}


def replicate_basename_rel(plan, k):
    return os.path.join("A2_4", "output", "uncertainty", plan, f"replicate_{k:02d}")


def replicate_paths(plan, k):
    base_rel = replicate_basename_rel(plan, k)
    return {
        "basename_rel": base_rel,
        "txt": os.path.join(PROJECT_ROOT, base_rel + ".txt"),
        "csv": os.path.join(PROJECT_ROOT, base_rel + ".csv"),
        "npz": os.path.join(PROJECT_ROOT, base_rel + ".npz"),
    }


def generate_replicate_inputs(plan, template_path):
    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()
    for k, seed in enumerate(SEEDS):
        paths = replicate_paths(plan, k)
        new_text = dvh_utils.rewrite_topas_for_replicate(
            template_text,
            seed=seed,
            histories=HISTORIES_PER_REPLICATE,
            output_basename=paths["basename_rel"],
        )
        os.makedirs(os.path.dirname(paths["txt"]), exist_ok=True)
        with open(paths["txt"], "w", encoding="utf-8") as f:
            f.write(new_text)


def build_masks():
    print("\nBuilding RTStruct masks ...")
    geom = load_ct_geometry(CT_DIR)
    masks, info = build_structure_masks(geom)
    print(f"Mask source: {info.get('source', 'unknown')}")
    for key in STRUCTURE_KEYS:
        print(f"  {key:<18s} voxels={len(masks.get(key, []))}")
    return masks


def run_one_plan(plan, template_path, masks, centres, edges, force, rescore):
    print()
    print("=" * 78)
    print(f"PLAN: {plan}")
    print("=" * 78)
    print(f"template            : {template_path}")
    print(f"replicates          : {N_REPLICATES}")
    print(f"histories/replicate : {HISTORIES_PER_REPLICATE:,}")

    if not os.path.isfile(template_path):
        print("Template not found. Run A2_4/sweep_beams.py first to create it.")
        return None

    generate_replicate_inputs(plan, template_path)

    replicate_dvhs = []
    for k, seed in enumerate(SEEDS):
        paths = replicate_paths(plan, k)

        if (not force) and (not rescore) and os.path.isfile(paths["npz"]):
            _, _, _, dvh = dvh_utils.load_replicate_npz(paths["npz"])
            replicate_dvhs.append(dvh)
            print(f"[{k + 1:02d}/{N_REPLICATES}] seed={seed}  (cached)")
            continue

        if rescore and os.path.isfile(paths["csv"]):
            print(f"[{k + 1:02d}/{N_REPLICATES}] seed={seed}  rescoring ... ",
                  end="", flush=True)
        else:
            t0 = time.time()
            print(f"[{k + 1:02d}/{N_REPLICATES}] seed={seed}  TOPAS ... ",
                  end="", flush=True)
            dvh_utils.run_topas(paths["txt"], project_root=PROJECT_ROOT)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.0f}s). Scoring DVH ... ", end="", flush=True)

        dose_dict = dvh_utils.load_topas_dose_csv(paths["csv"])
        plan_dose_max, dvh = dvh_utils.compute_replicate_dvhs(
            dose_dict, masks, STRUCTURE_KEYS, edges)
        dvh_utils.save_replicate_npz(
            paths["npz"], seed, plan_dose_max, centres, dvh)
        replicate_dvhs.append(dvh)
        print(f"plan_max={plan_dose_max:.3e} Gy")

    agg = dvh_utils.aggregate_replicates(
        replicate_dvhs, STRUCTURE_KEYS, centres)
    agg_path = os.path.join(UNCERT_ROOT, plan, "aggregated.npz")
    dvh_utils.save_aggregated_npz(agg_path, centres, agg)
    print(f"Aggregated: {agg_path}")
    return agg


def main():
    force   = FORCE_FLAG   in sys.argv
    rescore = RESCORE_FLAG in sys.argv
    os.makedirs(UNCERT_ROOT, exist_ok=True)

    print("=" * 78)
    print("A2_4 — MULTI-FIELD PHOTON MONTE CARLO UNCERTAINTY")
    print("=" * 78)
    print(f"Replicates/plan     : {N_REPLICATES}")
    print(f"Histories/replicate : {HISTORIES_PER_REPLICATE:,}")
    print(f"Seeds               : {SEEDS}")
    if force:
        print("Mode                : FORCE (re-run TOPAS + rescore)")
    elif rescore:
        print("Mode                : RESCORE (recompute DVHs from existing CSVs)")

    masks = None
    centres, edges = dvh_utils.common_dose_grid()

    results = {}
    for plan, template_path in PLANS.items():
        if masks is None:
            masks = build_masks()
        agg = run_one_plan(plan, template_path, masks, centres, edges,
                           force, rescore)
        if agg is not None:
            results[plan] = agg

    print()
    print("=" * 96)
    print("ENSEMBLE SUMMARY (mean ± 1σ across replicates, dose in % of plan max)")
    print("=" * 96)
    hdr = (f'{"Plan":<7s} {"Structure":<13s} '
           f'{"D95":>14s} {"D50":>14s} {"D02":>14s} {"Dmax":>14s}')
    print(hdr)
    print("-" * len(hdr))
    for plan, agg in results.items():
        for key in STRUCTURE_KEYS:
            stats = agg[key]

            def fmt(dname):
                m = stats.get(f"{dname}_mean", float("nan"))
                s = stats.get(f"{dname}_std",  float("nan"))
                if not np.isfinite(m):
                    return "      —       "
                return f"{m:6.2f} ± {s:4.2f}"
            print(f'{plan:<7s} {STRUCT_DISPLAY.get(key, key):<13s} '
                  f'{fmt("D95"):>14s} {fmt("D50"):>14s} {fmt("D02"):>14s} '
                  f'{fmt("Dmax"):>14s}')
        print()

    print("Done.")
    print("Next: python3 A2_4/analyse_dvh_multiplan_uncertainty.py")


if __name__ == "__main__":
    main()
