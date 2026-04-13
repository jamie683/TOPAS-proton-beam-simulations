"""
Shared DVH utilities for photon plan uncertainty quantification.

Used by:
    A2_3/run_uncertainty.py
    A2_3/analyse_dvh_uncertainty.py
    A2_4/run_uncertainty.py
    A2_4/analyse_dvh_multiplan_uncertainty.py

Responsibilities:
    1. Rewrite TOPAS input templates for seeded Monte Carlo replicates,
       preserving the relative split of histories between beams.
    2. Parse TOPAS voxel-level dose CSVs ("i,j,k,dose" rows).
    3. Build per-structure cumulative DVH curves on a common dose grid
       normalised to the replicate's own maximum dose.
    4. Aggregate multiple seed replicates into mean, 1-sigma, min and max
       curves per structure, plus mean-and-sigma of D95 / D50 / D02.
    5. Save / load replicate and aggregated DVHs as NPZ for caching.
    6. Plot DVH mean curves with a matching-colour shaded band.
"""

import os
import re
import subprocess

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401 — imported so callers can pass axes
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"matplotlib is required for dvh_utils: {exc}")


# ------------------------------------------------------------------
# Dose grid configuration
# ------------------------------------------------------------------
DVH_N_BINS = 200          # number of dose bins on the normalised grid
DVH_DOSE_MAX_PCT = 105.0  # upper bound of the grid in % of plan max


def common_dose_grid(n_bins=DVH_N_BINS, dose_max_pct=DVH_DOSE_MAX_PCT):
    """Return (centres_pct, edges_pct) for the common DVH dose grid."""
    edges = np.linspace(0.0, dose_max_pct, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, edges


# ------------------------------------------------------------------
# TOPAS dose CSV loader
# ------------------------------------------------------------------
def load_topas_dose_csv(csv_path):
    """Parse a TOPAS CSV with rows ``i, j, k, dose``.

    Comment lines (``#``) and blank lines are ignored. Rows with fewer than
    four comma-separated fields are skipped. Returns ``dict[(i,j,k) -> dose]``.
    """
    dose = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split(",")
            if len(parts) < 4:
                continue
            try:
                key = (int(parts[0]), int(parts[1]), int(parts[2]))
                dose[key] = float(parts[3])
            except (ValueError, TypeError):
                continue
    if not dose:
        raise ValueError(f"No voxel dose rows found in {csv_path}")
    return dose


def structure_dose_vector(dose_dict, mask):
    """Extract a 1-D dose array for voxels ``(i,j,k)`` present in ``mask``."""
    vals = [dose_dict[k] for k in mask if k in dose_dict]
    return np.asarray(vals, dtype=np.float64)


# ------------------------------------------------------------------
# Per-voxel replicate averaging
# ------------------------------------------------------------------
def average_replicate_dose_grids(csv_paths):
    """Load N replicate TOPAS dose CSVs and return (mean_dict, std_dict, n).

    Each CSV is parsed into a ``(i,j,k) -> dose`` dict; the union of all keys
    forms the voxel set. Voxels missing from a replicate are treated as zero
    dose (TOPAS only writes non-zero scoring voxels, so a missing voxel is
    statistically the same as a voxel whose ``n_hist`` samples landed at
    exactly zero dose — which for the purposes of per-voxel uncertainty is a
    fair approximation given our per-replicate history count).

    Returns two dicts in the same ``(i,j,k) -> float`` layout as
    ``load_topas_dose_csv`` so existing plot/DVH code can consume them
    directly, plus the number of replicates actually loaded.
    """
    csv_paths = [p for p in csv_paths if os.path.isfile(p)]
    n = len(csv_paths)
    if n == 0:
        return {}, {}, 0

    all_dicts = [load_topas_dose_csv(p) for p in csv_paths]
    all_keys = set()
    for d in all_dicts:
        all_keys.update(d.keys())

    mean_dict = {}
    std_dict  = {}
    for key in all_keys:
        vals = np.fromiter((d.get(key, 0.0) for d in all_dicts),
                           dtype=np.float64, count=n)
        mean_dict[key] = float(vals.mean())
        if n > 1:
            std_dict[key] = float(vals.std(ddof=1))
        else:
            std_dict[key] = 0.0
    return mean_dict, std_dict, n


# ------------------------------------------------------------------
# Cumulative DVH construction
# ------------------------------------------------------------------
def cumulative_dvh_percent(dose_vector_abs, plan_dose_max, edges_pct):
    """Cumulative DVH (% volume receiving at least each dose bin)."""
    n = int(dose_vector_abs.size)
    out = np.zeros(len(edges_pct) - 1, dtype=np.float64)
    if n == 0 or plan_dose_max <= 0:
        return out
    dose_pct = dose_vector_abs / plan_dose_max * 100.0
    hist, _ = np.histogram(dose_pct, bins=edges_pct)
    # Reverse cumsum: voxels receiving >= left edge of each bin.
    cum = np.cumsum(hist[::-1])[::-1]
    out[:] = 100.0 * cum / n
    return out


def compute_replicate_dvhs(dose_dict, masks, structure_keys, edges_pct):
    """Return (plan_dose_max, dict[structure -> dvh_curve]) for one replicate.

    The plan_dose_max is the maximum over all voxels in the dose grid
    (not only those inside any structure), so that DVHs for every structure
    share the same normalisation reference within the replicate.
    """
    if not dose_dict:
        return 0.0, {k: np.zeros(len(edges_pct) - 1) for k in structure_keys}
    dose_values = np.fromiter(dose_dict.values(), dtype=np.float64)
    plan_dose_max = float(dose_values.max()) if dose_values.size else 0.0
    dvh = {}
    for key in structure_keys:
        mask = masks.get(key, set())
        vec = structure_dose_vector(dose_dict, mask)
        dvh[key] = cumulative_dvh_percent(vec, plan_dose_max, edges_pct)
    return plan_dose_max, dvh


# ------------------------------------------------------------------
# DVH interpolation helper (used for D95/D50/D02)
# ------------------------------------------------------------------
def dose_at_volume(dose_pct, volume_pct, target_vol):
    """Interpolate the dose at which the cumulative DVH crosses ``target_vol``.

    For ``target_vol <= 0`` (Dmax), returns the highest dose bin where the
    cumulative volume is still above zero.  Returns ``float('nan')`` if the
    curve never reaches the target volume.
    """
    if target_vol <= 0:
        nonzero = np.where(np.asarray(volume_pct) > 0)[0]
        if len(nonzero) == 0:
            return float("nan")
        return float(dose_pct[nonzero[-1]])
    above = volume_pct >= target_vol
    if not np.any(above):
        return float("nan")
    idx = int(np.where(above)[0][-1])
    if idx >= len(dose_pct) - 1:
        return float(dose_pct[idx])
    v0, v1 = volume_pct[idx], volume_pct[idx + 1]
    d0, d1 = dose_pct[idx], dose_pct[idx + 1]
    if v0 == v1:
        return float(d0)
    frac = (target_vol - v0) / (v1 - v0)
    return float(d0 + frac * (d1 - d0))


# ------------------------------------------------------------------
# Ensemble aggregation
# ------------------------------------------------------------------
_D_TARGETS = (("D95", 95.0), ("D50", 50.0), ("D02", 2.0), ("Dmax", 0.0))


def aggregate_replicates(replicate_dvhs, structure_keys, dose_centres_pct):
    """Aggregate a list of replicate DVH dicts into mean / std / min / max.

    Each entry of ``replicate_dvhs`` is the ``dvh`` dict returned by
    ``compute_replicate_dvhs``. Per-replicate D95 / D50 / D02 values are also
    computed from each replicate's own curve and stored as mean ± 1-sigma.

    Returned dict layout::

        {
            "tumour": {
                "mean": <curve>, "std": <curve>, "min": <curve>, "max": <curve>,
                "n":     <int>,
                "D95_mean": <float>, "D95_std": <float>,
                "D50_mean": <float>, "D50_std": <float>,
                "D02_mean": <float>, "D02_std": <float>,
            },
            ...
        }
    """
    n = len(replicate_dvhs)
    if n == 0:
        raise ValueError("No replicates to aggregate.")
    agg = {}
    for key in structure_keys:
        stacked = np.stack([rep[key] for rep in replicate_dvhs], axis=0)
        entry = {
            "mean": stacked.mean(axis=0),
            "std":  stacked.std(axis=0, ddof=1) if n > 1 else np.zeros(stacked.shape[1]),
            "min":  stacked.min(axis=0),
            "max":  stacked.max(axis=0),
            "n":    n,
        }
        for dname, target_vol in _D_TARGETS:
            vals = []
            for rep in replicate_dvhs:
                vals.append(dose_at_volume(dose_centres_pct, rep[key], target_vol))
            arr = np.asarray(vals, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                entry[f"{dname}_mean"] = float(finite.mean())
                entry[f"{dname}_std"]  = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            else:
                entry[f"{dname}_mean"] = float("nan")
                entry[f"{dname}_std"]  = float("nan")
        agg[key] = entry
    return agg


# ------------------------------------------------------------------
# NPZ caching
# ------------------------------------------------------------------
def save_replicate_npz(path, seed, plan_dose_max, dose_centres_pct, dvh):
    """Cache a single replicate's DVH curves to disk."""
    payload = {
        "__seed":            np.int64(seed),
        "__plan_dose_max":   np.float64(plan_dose_max),
        "__dose_centres_pct": np.asarray(dose_centres_pct, dtype=np.float64),
    }
    for key, curve in dvh.items():
        payload[key] = np.asarray(curve, dtype=np.float64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)


def load_replicate_npz(path):
    """Return (seed, plan_dose_max, centres, dvh_dict)."""
    data = np.load(path)
    seed = int(data["__seed"])
    plan_dose_max = float(data["__plan_dose_max"])
    centres = data["__dose_centres_pct"]
    dvh = {k: data[k] for k in data.files if not k.startswith("__")}
    return seed, plan_dose_max, centres, dvh


def save_aggregated_npz(path, dose_centres_pct, agg):
    """Cache an aggregated DVH ensemble to disk."""
    payload = {"__dose_centres_pct": np.asarray(dose_centres_pct, dtype=np.float64)}
    for key, stats in agg.items():
        for field, val in stats.items():
            arr_key = f"{key}__{field}"
            if isinstance(val, (int, np.integer)):
                payload[arr_key] = np.int64(val)
            elif isinstance(val, (float, np.floating)):
                payload[arr_key] = np.float64(val)
            else:
                payload[arr_key] = np.asarray(val, dtype=np.float64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **payload)


def load_aggregated_npz(path):
    """Return (dose_centres_pct, dict[structure -> stats])."""
    data = np.load(path)
    centres = data["__dose_centres_pct"]
    agg = {}
    for k in data.files:
        if k == "__dose_centres_pct":
            continue
        if "__" not in k:
            continue
        struct, field = k.split("__", 1)
        agg.setdefault(struct, {})
        val = data[k]
        if val.shape == ():
            agg[struct][field] = val.item()
        else:
            agg[struct][field] = np.asarray(val)
    return centres, agg


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_dvh_with_band(ax, dose_centres_pct, mean, std, color, label,
                       linestyle="-", linewidth=1.8, alpha=0.25):
    """Plot a mean DVH curve with a matching-colour ±1σ shaded band.

    The band is clipped to the physical range [0, 100] % volume so it never
    extends below the axis or above full volume.
    """
    mean = np.asarray(mean, dtype=np.float64)
    std  = np.asarray(std,  dtype=np.float64)
    lower = np.clip(mean - std, 0.0, 100.0)
    upper = np.clip(mean + std, 0.0, 100.0)
    ax.fill_between(dose_centres_pct, lower, upper,
                    color=color, alpha=alpha, linewidth=0)
    ax.plot(dose_centres_pct, mean,
            color=color, linestyle=linestyle, linewidth=linewidth, label=label)


# ------------------------------------------------------------------
# TOPAS execution
# ------------------------------------------------------------------
TOPAS_EXE_DEFAULT = "/home/jamie/shellScripts/topas"


def run_topas(param_file, project_root, topas_exe=TOPAS_EXE_DEFAULT, timeout=7200):
    """Run TOPAS on a single parameter file. Raises ``RuntimeError`` on failure."""
    rel = os.path.relpath(param_file, project_root)
    result = subprocess.run(
        [topas_exe, rel],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-1000:]
        raise RuntimeError(f"TOPAS failed for {rel}:\n{tail}")


# ------------------------------------------------------------------
# Template rewriting (histories / seed / output type / output file)
# ------------------------------------------------------------------
_HISTORY_RE = re.compile(
    r"^(\s*i:So/\S+/NumberOfHistoriesInRun\s*=\s*)(\d+)", re.M)
_SEED_RE = re.compile(r"^\s*i:Ts/Seed\s*=.*$", re.M)
_OUTTYPE_RE = re.compile(
    r'^(\s*s:Sc/\S+/OutputType\s*=\s*)"[^"]*"', re.M)
_OUTFILE_RE = re.compile(
    r'^(\s*s:Sc/\S+/OutputFile\s*=\s*)"[^"]*"', re.M)
_THREADS_RE = re.compile(
    r"^(\s*i:Ts/NumberOfThreads\s*=\s*\d+)", re.M)


def rewrite_topas_for_replicate(template_text, seed, histories, output_basename):
    """Return a new TOPAS input string configured for one seeded replicate.

    Changes made to the template:
      * The total ``NumberOfHistoriesInRun`` across all beams is rescaled to
        ``histories``, preserving the relative per-beam split.
      * ``i:Ts/Seed`` is set to ``seed`` (inserted after ``NumberOfThreads``
        if no seed line already exists).
      * All scorer ``OutputType`` entries are forced to ``"csv"``.
      * All scorer ``OutputFile`` entries are set to ``output_basename``
        (no extension — TOPAS appends ``.csv`` automatically).

    All other lines are preserved verbatim, so per-beam geometry, weights,
    apertures and scorer names stay exactly as produced by the upstream
    pipeline.
    """
    text = template_text

    matches = list(_HISTORY_RE.finditer(text))
    if not matches:
        raise ValueError(
            "Template has no 'i:So/*/NumberOfHistoriesInRun' line — "
            "cannot rewrite for a replicate.")

    old_counts = [int(m.group(2)) for m in matches]
    old_total = sum(old_counts)
    if old_total <= 0:
        raise ValueError("Template specifies zero total histories.")

    # Rescale per-beam counts while preserving their relative weights.
    scale = float(histories) / float(old_total)
    new_counts = [max(1, int(round(c * scale))) for c in old_counts]
    drift = int(histories) - sum(new_counts)
    if drift != 0 and new_counts:
        new_counts[0] = max(1, new_counts[0] + drift)

    count_iter = iter(new_counts)

    def _replace_hist(match):
        return f"{match.group(1)}{next(count_iter)}"
    text = _HISTORY_RE.sub(_replace_hist, text)

    # Seed — replace existing line if present, otherwise insert after threads.
    seed_line = f"i:Ts/Seed                       = {int(seed)}"
    if _SEED_RE.search(text):
        text = _SEED_RE.sub(seed_line, text)
    else:
        text = _THREADS_RE.sub(lambda m: f"{m.group(1)}\n{seed_line}", text,
                               count=1)

    # Scorer output: force CSV and point at the replicate basename.
    text = _OUTTYPE_RE.sub(lambda m: f'{m.group(1)}"csv"', text)
    text = _OUTFILE_RE.sub(lambda m: f'{m.group(1)}"{output_basename}"', text)

    return text
