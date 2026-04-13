"""
A2_8 — Side-by-side comparison of analytical PBS dose vs TOPAS Monte Carlo.

Requires:
    A2_8/output/dose_pbs_analytical.csv  (from pbs_proton.py with SKIP_TOPAS=True)
    A2_8/output/NEW/dose_pbs_patient.csv (from running dose_pbs_patient.txt in TOPAS)

Produces:
    A2_8/output/pbs_analytical_vs_mc.png  — patient slice comparison
    A2_8/output/dvh_analytical_vs_mc.png  — DVH overlay

Usage:
    python3 A2_8/plot_analytical_vs_mc.py
"""

import os
import sys

import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "A2_4"))
sys.path.insert(0, SCRIPT_DIR)
import dvh_utils  # noqa: E402
from sweep_beams import load_ct_geometry  # noqa: E402
from pbs_proton import build_scoring_masks  # noqa: E402

CT_DIR = os.path.join(PROJECT_ROOT, "CTData")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

ANALYTICAL_CSV = os.path.join(OUTPUT_DIR, "dose_pbs_analytical.csv")
MC_CSV = os.path.join(OUTPUT_DIR, "NEW", "dose_pbs_patient.csv")

TUMOUR_Z = 0.0
CONTOUR_NUDGE_PX = 0.5
PRESCRIPTION_GY = 60.0

CONTOUR_COLOURS = {
    "GTVp": "red", "GTV": "red", "PTV": "magenta",
    "Lung_R": "orange", "Lung_L": "cyan", "Heart": "blue",
    "SpinalCord": "green", "Body": "grey", "BODY": "grey",
}
CONTOUR_ORDER = [
    "Body", "BODY", "Lung_R", "Lung_L", "Heart",
    "SpinalCord", "PTV", "GTVp", "GTV",
]

STRUCTURE_KEYS = ("tumour", "ptv", "lung_r", "lung_l", "heart", "cord", "body")
STRUCTURE_DISPLAY = {
    "tumour": "GTVp", "ptv": "PTV", "lung_r": "Right Lung",
    "lung_l": "Left Lung", "heart": "Heart", "cord": "Spinal Cord",
    "body": "Body",
}
STRUCTURE_COLORS = {
    "tumour": "red", "ptv": "magenta", "lung_r": "orange",
    "lung_l": "olive", "heart": "blue", "cord": "green", "body": "grey",
}


# ------------------------------------------------------------------
# Helpers (matching plot_uncertainty_slices.py)
# ------------------------------------------------------------------
def _find_rtstruct():
    for fname in sorted(os.listdir(CT_DIR)):
        if not fname.lower().endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(os.path.join(CT_DIR, fname),
                                 stop_before_pixels=True)
            if getattr(ds, "Modality", "") == "RTSTRUCT":
                return os.path.join(CT_DIR, fname)
        except Exception:
            pass
    return None


def load_contour_polygons(geom, slice_iz):
    rtstruct_path = _find_rtstruct()
    if not rtstruct_path:
        return {}
    ds = pydicom.dcmread(rtstruct_path)
    roi_map = {}
    for roi in getattr(ds, "StructureSetROISequence", []):
        roi_map[int(roi.ROINumber)] = str(roi.ROIName)
    contours = {}
    for roi_contour in getattr(ds, "ROIContourSequence", []):
        roi_number = int(getattr(roi_contour, "ReferencedROINumber", -1))
        roi_name = roi_map.get(roi_number)
        if not roi_name or not hasattr(roi_contour, "ContourSequence"):
            continue
        for contour in roi_contour.ContourSequence:
            data = np.asarray(getattr(contour, "ContourData", []), dtype=float)
            if data.size < 9:
                continue
            pts = data.reshape(-1, 3)
            z_mean = float(np.mean(pts[:, 2]))
            iz = int(np.argmin(np.abs(np.array(geom["slice_zs"]) - z_mean)))
            if iz != slice_iz:
                continue
            contours.setdefault(roi_name, []).append(pts[:, :2])
    return contours


def load_ct_slice_hu(geom, z_mm):
    iz = int(np.argmin(np.abs(np.array(geom["slice_zs"]) - z_mm)))
    target_z = geom["slice_zs"][iz]
    for fname in sorted(os.listdir(CT_DIR)):
        if not fname.lower().endswith(".dcm"):
            continue
        ds = pydicom.dcmread(os.path.join(CT_DIR, fname))
        if getattr(ds, "Modality", "") != "CT":
            continue
        if abs(float(ds.ImagePositionPatient[2]) - target_z) < 0.1:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            return ds.pixel_array.astype(float) * slope + intercept, iz
    raise FileNotFoundError(f"No CT slice near Z={z_mm} mm")


def slice_extent(geom):
    dx, dy = geom["dx"], geom["dy"]
    return [geom["x0"] - dx / 2, geom["x0"] + (geom["cols"] - 0.5) * dx,
            geom["y0"] - dy / 2, geom["y0"] + (geom["rows"] - 0.5) * dy]


def dose_dict_to_slice_2d(dose_dict, geom, slice_iz):
    img = np.zeros((geom["rows"], geom["cols"]), dtype=np.float64)
    for (ix, iy, iz), d in dose_dict.items():
        if iz == slice_iz and 0 <= ix < geom["cols"] and 0 <= iy < geom["rows"]:
            img[iy, ix] = d
    return img


def draw_contours(ax, contour_polys, geom):
    shift = np.array([geom["dx"] * CONTOUR_NUDGE_PX,
                      geom["dy"] * CONTOUR_NUDGE_PX])
    drawn = set()
    handles = []
    for name in CONTOUR_ORDER:
        if name in contour_polys and name not in drawn:
            colour = CONTOUR_COLOURS.get(name, "white")
            for poly_xy in contour_polys[name]:
                ax.add_patch(MplPolygon(
                    poly_xy + shift, closed=True, fill=False,
                    edgecolor=colour, linewidth=1.3))
            handles.append(Line2D([], [], color=colour, linewidth=1.3, label=name))
            drawn.add(name)
    for name, polys in contour_polys.items():
        if name not in drawn:
            colour = CONTOUR_COLOURS.get(name, "white")
            for poly_xy in polys:
                ax.add_patch(MplPolygon(
                    poly_xy + shift, closed=True, fill=False,
                    edgecolor=colour, linewidth=1.3))
            handles.append(Line2D([], [], color=colour, linewidth=1.3, label=name))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7,
                  frameon=True, framealpha=0.8, edgecolor="grey")


# ------------------------------------------------------------------
# Plot 1: Patient slice comparison
# ------------------------------------------------------------------
def plot_slice_comparison(analytical_dict, mc_dict, geom, contour_polys,
                          hu_image, slice_iz, out_path):
    extent = slice_extent(geom)
    target_z = geom["slice_zs"][slice_iz]

    ana_2d = dose_dict_to_slice_2d(analytical_dict, geom, slice_iz)
    mc_2d = dose_dict_to_slice_2d(mc_dict, geom, slice_iz)

    vmax = max(ana_2d.max(), mc_2d.max())
    if vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), constrained_layout=True)

    for ax, dose_2d, title in [
        (axes[0], ana_2d, "Analytical pencil beam model"),
        (axes[1], mc_2d,  "TOPAS Monte Carlo (500k histories)"),
    ]:
        ax.imshow(hu_image, cmap="gray", extent=extent, origin="lower",
                  vmin=-400, vmax=400, aspect="equal")
        masked = np.ma.masked_where(dose_2d < 0.05 * vmax, dose_2d)
        im = ax.imshow(masked, cmap="jet", extent=extent, origin="lower",
                       alpha=0.55, vmin=0, vmax=vmax, aspect="equal")
        draw_contours(ax, contour_polys, geom)
        ax.invert_yaxis()
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85, label="Dose (Gy)")

    fig.suptitle(f"PBS Dose Distribution — Analytical vs Monte Carlo\n"
                 f"Axial slice Z = {target_z:.1f} mm, both normalised to "
                 f"{PRESCRIPTION_GY:.0f} Gy GTV mean", fontsize=13)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ------------------------------------------------------------------
# Plot 2: DVH comparison
# ------------------------------------------------------------------
def plot_dvh_comparison(analytical_dict, mc_dict, masks, plan_max, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    for key in STRUCTURE_KEYS:
        mask = masks.get(key, set())
        if not mask:
            continue
        color = STRUCTURE_COLORS[key]
        display = STRUCTURE_DISPLAY[key]

        for dose_dict, ls, lw, suffix in [
            (analytical_dict, "-",  1.8, "analytical"),
            (mc_dict,         "--", 1.3, "TOPAS MC"),
        ]:
            doses = np.array([dose_dict.get(v, 0.0) for v in mask])
            if doses.size == 0 or doses.max() <= 0:
                continue
            dose_pct = doses / plan_max * 100.0
            bins = np.linspace(0, 115, 201)
            centres = 0.5 * (bins[:-1] + bins[1:])
            hist, _ = np.histogram(dose_pct, bins=bins)
            cum = np.cumsum(hist[::-1])[::-1]
            vol_pct = 100.0 * cum / doses.size
            ax.plot(centres, vol_pct, color=color, linestyle=ls,
                    linewidth=lw, label=f"{display} ({suffix})")

    ax.set_xlabel("Dose (% of plan max)", fontsize=12)
    ax.set_ylabel("Volume (%)", fontsize=12)
    ax.set_title("PBS DVH — Analytical Model vs TOPAS Monte Carlo", fontsize=13)
    ax.set_xlim(0, 115)
    ax.set_ylim(0, 105)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ------------------------------------------------------------------
def main():
    if not os.path.isfile(ANALYTICAL_CSV):
        print(f"Analytical CSV not found: {ANALYTICAL_CSV}")
        print("Run: python3 A2_8/pbs_proton.py  (with SKIP_TOPAS = True)")
        return
    if not os.path.isfile(MC_CSV):
        print(f"MC CSV not found: {MC_CSV}")
        print("Run the TOPAS PBS template manually, place CSV in A2_8/output/NEW/")
        return

    print("Loading CT geometry ...")
    geom = load_ct_geometry(CT_DIR)
    masks = build_scoring_masks(geom)

    print("Loading analytical dose ...")
    analytical_dict = dvh_utils.load_topas_dose_csv(ANALYTICAL_CSV)
    print(f"  {len(analytical_dict)} voxels, max = {max(analytical_dict.values()):.2f} Gy")

    print("Loading TOPAS MC dose ...")
    mc_dict = dvh_utils.load_topas_dose_csv(MC_CSV)
    # Normalise MC to 60 Gy at GTV mean
    mc_tumour = np.array([mc_dict.get(k, 0.0) for k in masks["tumour"]])
    mc_scale = PRESCRIPTION_GY / mc_tumour.mean()
    mc_dict = {k: v * mc_scale for k, v in mc_dict.items()}
    print(f"  MC scale factor: {mc_scale:.2e}")

    print("Loading CT slice and contours ...")
    hu_image, slice_iz = load_ct_slice_hu(geom, TUMOUR_Z)
    contour_polys = load_contour_polygons(geom, slice_iz)

    out_slice = os.path.join(OUTPUT_DIR, "pbs_analytical_vs_mc.png")
    plot_slice_comparison(analytical_dict, mc_dict, geom, contour_polys,
                          hu_image, slice_iz, out_slice)

    plan_max = max(analytical_dict.values())
    print(f"  Plan max (analytical): {plan_max:.2f} Gy")

    out_dvh = os.path.join(OUTPUT_DIR, "dvh_analytical_vs_mc.png")
    plot_dvh_comparison(analytical_dict, mc_dict, masks, plan_max, out_dvh)

    # Print comparison table (% of plan max)
    print()
    print("=" * 85)
    print(f"DVH METRICS: ANALYTICAL vs TOPAS MC  (% of plan max = {plan_max:.2f} Gy)")
    print("=" * 85)
    print(f'{"Structure":<14s} {"Model":>5s} {"D95":>8s} {"D50":>8s} '
          f'{"D02":>8s} {"Mean":>8s} {"Max":>8s}')
    print("-" * 65)
    for key in STRUCTURE_KEYS:
        mask = masks.get(key, set())
        if not mask:
            continue
        display = STRUCTURE_DISPLAY[key]
        for dose_dict, label in [(analytical_dict, "Ana"), (mc_dict, "MC")]:
            doses = np.array([dose_dict.get(v, 0.0) for v in mask])
            pct = doses / plan_max * 100.0
            d95 = np.percentile(pct, 5)
            d50 = np.percentile(pct, 50)
            d02 = np.percentile(pct, 98)
            mean = pct.mean()
            mx = pct.max()
            name = display if label == "Ana" else ""
            print(f'{name:<14s} {label:>5s} {d95:7.1f}% {d50:7.1f}% '
                  f'{d02:7.1f}% {mean:7.1f}% {mx:7.1f}%')
        print()

    print("Done.")


if __name__ == "__main__":
    main()
