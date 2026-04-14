"""
A2_4 — per-plan axial slice plots of the mean MC dose and relative 1-sigma.

For each photon plan (1-field + 2/3/4-beam) the driver:

    1. Locates the seed-replicate CSVs produced by ``A2_3/run_uncertainty.py``
       and ``A2_4/run_uncertainty.py``.
    2. Averages the per-voxel dose across the N replicates to produce a
       noise-reduced "best estimate" dose cube (equivalent in variance to a
       single run with N × per-replicate histories) along with a per-voxel
       standard deviation.
    3. Loads the CT slice nearest the tumour centre and overlays:
           (a) the mean absolute dose (jet, masked at 5 % of slice max),
           (b) the relative statistical uncertainty σ / μ in percent, masked
               to voxels above 20 % of the slice's mean-dose maximum so the
               displayed ratio is well defined.
    4. Draws structure contours (GTVp, PTV, lungs, heart, cord, body) from
       the RTStruct file with colours matching the A2_6 patient dose overlay.
    5. Writes one two-panel PNG per plan under ``A2_4/output/``.

Usage:
    python3 A2_4/plot_uncertainty_slices.py
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

sys.path.insert(0, SCRIPT_DIR)
import dvh_utils  # noqa: E402
from sweep_beams import load_ct_geometry  # noqa: E402


CT_DIR = os.path.join(PROJECT_ROOT, "CTData")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

N_REPLICATES = 10
TUMOUR_Z = 0.0           # mm — centre of the slice we want to render
REL_SIGMA_MASK_PCT = 20.0  # only show σ/μ above this fraction of slice max
REL_SIGMA_VMAX_PCT = 10.0  # colourbar upper bound for σ/μ, % (clip the tail)

# Sub-pixel contour nudge (in units of pixels).  Compensates for the common
# half-pixel misregistration between the CT ImagePositionPatient convention
# and the RTStruct coordinate frame.  Positive = shift contour right / down.
CONTOUR_NUDGE_PX = 0.5

PLANS = [
    # (display_name, output_suffix, csv path template)
    ("1-field", "1_field",
     os.path.join(PROJECT_ROOT, "A2_3", "output", "uncertainty",
                  "replicate_{k:02d}.csv")),
    ("2-beam",  "2_beam",
     os.path.join(SCRIPT_DIR, "output", "uncertainty", "2beam",
                  "replicate_{k:02d}.csv")),
    ("3-beam",  "3_beam",
     os.path.join(SCRIPT_DIR, "output", "uncertainty", "3beam",
                  "replicate_{k:02d}.csv")),
    ("4-beam",  "4_beam",
     os.path.join(SCRIPT_DIR, "output", "uncertainty", "4beam",
                  "replicate_{k:02d}.csv")),
    ("proton 1-beam", "proton_1beam",
     os.path.join(PROJECT_ROOT, "A2_5", "output", "uncertainty",
                  "replicate_{k:02d}.csv")),
    ("SOBP (initial narrow beam)", "sobp_initial",
     os.path.join(PROJECT_ROOT, "A2_6", "outputs", "uncertainty_initial",
                  "replicate_{k:02d}.csv")),
    ("SOBP",    "sobp",
     os.path.join(PROJECT_ROOT, "A2_6", "outputs", "uncertainty",
                  "replicate_{k:02d}.csv")),
    ("PBS",     "pbs",
     os.path.join(PROJECT_ROOT, "A2_8", "output", "uncertainty",
                  "replicate_{k:02d}.csv")),
]

# RTStruct ROI name → contour colour.  Matches the A2_6 patient overlay
# palette and adds left-lung / PTV.  Any ROI not listed here is drawn white.
CONTOUR_COLOURS = {
    "GTVp":       "red",
    "GTV":        "red",
    "PTV":        "magenta",
    "Lung_R":     "orange",
    "Lung_L":     "cyan",
    "Heart":      "blue",
    "SpinalCord": "green",
    "Body":       "grey",
    "BODY":       "grey",
    "External":   "grey",
    "EXTERNAL":   "grey",
}
# Drawing order — outermost first so inner contours sit on top.
CONTOUR_ORDER = [
    "Body", "BODY", "External", "EXTERNAL",
    "Lung_R", "Lung_L",
    "Heart",
    "SpinalCord",
    "PTV",
    "GTVp", "GTV",
]


# ------------------------------------------------------------------
# RTStruct vector contours (drawn as matplotlib patches — no
# rasterisation, so alignment is pixel-perfect like VICTORIA)
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


def load_rtstruct_contour_polygons(geom, slice_iz):
    """Load RTStruct and return {roi_name: [array(N,2), ...]} for one slice.

    Each value is a list of closed polygon vertex arrays in DICOM patient
    coordinates (mm).  These are drawn directly as matplotlib patches so
    there is zero rasterisation error.
    """
    rtstruct_path = _find_rtstruct()
    if not rtstruct_path:
        print("  [warn] no RTSTRUCT file found — contours will be missing")
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


# ------------------------------------------------------------------
# CT helpers
# ------------------------------------------------------------------
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


def dose_dict_to_slice_2d(dose_dict, geom, slice_iz):
    img = np.zeros((geom["rows"], geom["cols"]), dtype=np.float64)
    for (ix, iy, iz), d in dose_dict.items():
        if iz == slice_iz and 0 <= ix < geom["cols"] and 0 <= iy < geom["rows"]:
            img[iy, ix] = d
    return img


def slice_extent(geom):
    """imshow extent — shift by half a pixel so voxel centres line up."""
    dx = geom["dx"]; dy = geom["dy"]
    x_min = geom["x0"] - dx / 2
    x_max = geom["x0"] + (geom["cols"] - 0.5) * dx
    y_min = geom["y0"] - dy / 2
    y_max = geom["y0"] + (geom["rows"] - 0.5) * dy
    return [x_min, x_max, y_min, y_max]


# ------------------------------------------------------------------
# Contour drawing (vector patches — pixel-perfect alignment)
# ------------------------------------------------------------------
def draw_contours(ax, contour_polys, geom):
    """Draw RTStruct polygon outlines directly as matplotlib patches."""
    shift = np.array([geom["dx"] * CONTOUR_NUDGE_PX,
                      geom["dy"] * CONTOUR_NUDGE_PX])
    drawn = set()
    legend_handles = []
    # Draw in defined order first …
    for name in CONTOUR_ORDER:
        if name in contour_polys and name not in drawn:
            colour = CONTOUR_COLOURS.get(name, "white")
            for poly_xy in contour_polys[name]:
                ax.add_patch(MplPolygon(
                    poly_xy + shift, closed=True, fill=False,
                    edgecolor=colour, linewidth=1.3))
            legend_handles.append(
                Line2D([], [], color=colour, linewidth=1.3, label=name))
            drawn.add(name)
    # … then any remaining ROIs.
    for name, polys in contour_polys.items():
        if name not in drawn:
            colour = CONTOUR_COLOURS.get(name, "white")
            for poly_xy in polys:
                ax.add_patch(MplPolygon(
                    poly_xy + shift, closed=True, fill=False,
                    edgecolor=colour, linewidth=1.3))
            legend_handles.append(
                Line2D([], [], color=colour, linewidth=1.3, label=name))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
                  frameon=True, framealpha=0.8, edgecolor="grey")


# ------------------------------------------------------------------
# Main two-panel plot
# ------------------------------------------------------------------
def plot_plan_slice(plan_name, mean_dict, std_dict, n_rep,
                    geom, contour_polys, out_path):
    hu_image, slice_iz = load_ct_slice_hu(geom, TUMOUR_Z)
    target_z = geom["slice_zs"][slice_iz]

    mean_2d = dose_dict_to_slice_2d(mean_dict, geom, slice_iz)
    std_2d  = dose_dict_to_slice_2d(std_dict,  geom, slice_iz)

    if mean_2d.max() <= 0:
        print(f"[skip] {plan_name}: dose slice at Z={target_z:.1f} mm is empty")
        return

    dose_max = float(mean_2d.max())
    mean_thresh = 0.01 * REL_SIGMA_MASK_PCT * dose_max
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_sigma_2d = np.where(mean_2d > mean_thresh,
                                100.0 * std_2d / mean_2d, np.nan)

    extent = slice_extent(geom)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0),
                             constrained_layout=True)

    # ---- Left panel: mean dose (% of slice max) ----------------------
    ax = axes[0]
    ax.imshow(hu_image, cmap="gray", extent=extent, origin="lower",
              vmin=-400, vmax=400, aspect="equal")
    mean_pct = mean_2d / dose_max * 100.0
    dose_masked = np.ma.masked_where(mean_pct < 5.0, mean_pct)
    im1 = ax.imshow(dose_masked, cmap="jet", extent=extent, origin="lower",
                    alpha=0.55, vmin=0, vmax=100.0, aspect="equal")
    cbar1 = fig.colorbar(im1, ax=ax, pad=0.02, shrink=0.85)
    cbar1.set_label("Mean dose (% of slice max)", fontsize=11)
    draw_contours(ax, contour_polys, geom)
    ax.invert_yaxis()
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_title(f"Mean dose — {plan_name}\n"
                 f"averaged over {n_rep} seeds, Z={target_z:.1f} mm",
                 fontsize=12)

    # ---- Right panel: relative 1-sigma ------------------------------
    ax = axes[1]
    ax.imshow(hu_image, cmap="gray", extent=extent, origin="lower",
              vmin=-400, vmax=400, aspect="equal")
    rel_masked = np.ma.masked_invalid(rel_sigma_2d)
    im2 = ax.imshow(rel_masked, cmap="magma", extent=extent, origin="lower",
                    alpha=0.70, vmin=0, vmax=REL_SIGMA_VMAX_PCT,
                    aspect="equal")
    cbar2 = fig.colorbar(im2, ax=ax, pad=0.02, shrink=0.85,
                         extend="max")
    cbar2.set_label(r"Relative 1$\sigma$ = $\sigma/\mu$ (%)", fontsize=11)
    draw_contours(ax, contour_polys, geom)
    ax.invert_yaxis()
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_title(f"Relative 1σ — {plan_name}\n"
                 f"masked to μ > {REL_SIGMA_MASK_PCT:.0f}% of slice max",
                 fontsize=12)

    # Console convergence summary.
    high_dose = rel_sigma_2d[np.isfinite(rel_sigma_2d)]
    if high_dose.size:
        q50 = float(np.percentile(high_dose, 50))
        q95 = float(np.percentile(high_dose, 95))
        print(f"  {plan_name}: median σ/μ = {q50:4.2f}%, "
              f"95-th pct = {q95:4.2f}%, voxels = {high_dose.size}")

    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading CT geometry ...")
    geom = load_ct_geometry(CT_DIR)

    print("Loading RTStruct contour polygons ...")
    _, slice_iz = load_ct_slice_hu(geom, TUMOUR_Z)
    contour_polys = load_rtstruct_contour_polygons(geom, slice_iz)
    print(f"  structures on slice iz={slice_iz}: "
          f"{sorted(contour_polys.keys())}")

    any_plan = False
    for display, suffix, path_template in PLANS:
        csv_paths = [path_template.format(k=k) for k in range(N_REPLICATES)]
        available = [p for p in csv_paths if os.path.isfile(p)]
        if not available:
            print(f"[skip] {display}: no replicate CSVs under "
                  f"{os.path.dirname(path_template)}")
            continue

        print(f"\n== {display}: averaging {len(available)} replicate CSVs ==")
        mean_dict, std_dict, n_rep = dvh_utils.average_replicate_dose_grids(
            available)
        out_path = os.path.join(
            OUTPUT_DIR, f"dose_slice_{suffix}_mean_and_relsigma.png")
        plot_plan_slice(display, mean_dict, std_dict, n_rep,
                        geom, contour_polys, out_path)
        any_plan = True

    if not any_plan:
        print("\nNo replicate CSVs found anywhere. Run the uncertainty "
              "drivers first:")
        print("  python3 A2_3/run_uncertainty.py")
        print("  python3 A2_4/run_uncertainty.py")
        return

    print("\nDone.")


if __name__ == "__main__":
    main()
