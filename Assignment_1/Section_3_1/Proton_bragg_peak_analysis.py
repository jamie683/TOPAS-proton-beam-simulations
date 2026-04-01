import re
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

TOPAS_EXE = "/home/jamie/shellScripts/topas"
BASE_TXT = Path("proton_water_bragg_peak.txt")

SEEDS = [1, 2, 3, 4, 5]

def replace_line(text: str, pattern: str, new_line: str) -> str:
    rgx = re.compile(pattern, re.MULTILINE)
    if not rgx.search(text):
        raise RuntimeError(f"Pattern not found: {pattern}")
    return rgx.sub(new_line, text, count=1)

def run_topas(param_file: Path):
    subprocess.run([TOPAS_EXE, str(param_file)], check=True)

def load_peak_depth_cm(csv_path: Path) -> float:
    data = np.loadtxt(csv_path, comments="#", delimiter=",")
    iz = data[:, 2]
    dose = data[:, 3]
    # from your header: Z bins are 0.05 cm each
    depth_cm = (iz + 0.5) * 0.05
    return float(depth_cm[np.argmax(dose)])

def peak_depth_parabolic(depth_cm: np.ndarray, y: np.ndarray) -> float:
    """Sub-bin peak estimate using a quadratic through (i-1,i,i+1) around argmax."""
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return float(depth_cm[i])

    x1, x2, x3 = depth_cm[i-1], depth_cm[i], depth_cm[i+1]
    y1, y2, y3 = y[i-1], y[i], y[i+1]

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if denom == 0:
        return float(x2)

    A = (x3*(y2 - y1) + x2*(y1 - y3) + x1*(y3 - y2)) / denom
    B = (x3**2*(y1 - y2) + x2**2*(y3 - y1) + x1**2*(y2 - y3)) / denom

    if A == 0:
        return float(x2)

    x_vertex = -B / (2*A)
    return float(x_vertex)

base_text = BASE_TXT.read_text()

peak_depths = []
curves = []

for seed in SEEDS:
    out_base = f"DoseZ_seed{seed:03d}"
    run_txt = Path(f"run_seed{seed:03d}.txt")
    out_csv = Path(f"{out_base}.csv")

    # Deletes old outputs
    for p in Path(".").glob(f"{out_base}.csv"):
        p.unlink()
    for p in Path(".").glob(f"{out_base}_*.csv"):
        p.unlink()

    txt = base_text
    txt = replace_line(txt, r"^i:Ts/Seed\s*=.*$", f"i:Ts/Seed = {seed}")
    txt = replace_line(txt, r'^s:Sc/Dose/OutputFile\s*=.*$', f's:Sc/Dose/OutputFile = "{out_base}"')
    run_txt.write_text(txt)

    print(f"Running seed {seed}...")
    run_topas(run_txt)

    if not out_csv.exists():
        raise RuntimeError(f"Expected output not created: {out_csv}")

    # Load curve + peak
    data = np.loadtxt(out_csv, comments="#", delimiter=",")
    iz = data[:, 2]
    dose = data[:, 3]
    depth_cm = (iz + 0.5) * 0.05

    peak_cm = peak_depth_parabolic(depth_cm, dose)
    peak_depths.append(peak_cm)
    curves.append((depth_cm, dose / np.max(dose)))

mean = float(np.mean(peak_depths))
std = float(np.std(peak_depths, ddof=1)) if len(peak_depths) > 1 else 0.0

print("\nPeak depth results (cm):")
for s, p in zip(SEEDS, peak_depths):
    print(f"  Seed {s:3d}: {p:.4f} cm")
print(f"\nMean ± SD: {mean:.4f} ± {std:.4f} cm  (n={len(SEEDS)})")

# Plot overlay
plt.figure(figsize=(9,5))
for (d, y), s in zip(curves, SEEDS):
    plt.plot(d, y, alpha=0.35, label=f"seed {s}")
plt.xlabel("Depth in water from entrance (cm)")
plt.ylabel("Normalised DoseToMedium")
plt.title("220 MeV proton Bragg peak in water (seed repeats)")
plt.grid(True)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig("section1_bragg_overlay.png", dpi=300)
plt.show()
