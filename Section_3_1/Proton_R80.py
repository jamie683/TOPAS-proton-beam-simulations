from pathlib import Path
import numpy as np

Z_BIN_WIDTH_CM = 0.05  # 40 cm / 800 bins

def load_depth_dose_from_csv(csv_path: Path):
    data = np.loadtxt(csv_path, comments="#", delimiter=",")
    iz = data[:, 2]
    dose = data[:, 3]
    depth_cm = (iz + 0.5) * Z_BIN_WIDTH_CM
    return depth_cm, dose

def distal_R(depth_cm: np.ndarray, dose: np.ndarray, level: float) -> float:
    y = dose / np.max(dose)
    i_peak = int(np.argmax(y))
    d = depth_cm[i_peak:]
    yy = y[i_peak:]

    idx = np.where((yy[:-1] >= level) & (yy[1:] < level))[0]
    if len(idx) == 0:
        return float("nan")

    i = idx[0]
    x0, x1 = d[i], d[i + 1]
    y0, y1 = yy[i], yy[i + 1]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))

def compute_Rs(depth_cm, dose):
    R80 = distal_R(depth_cm, dose, level=0.8)
    R90 = distal_R(depth_cm, dose, level=0.9)
    return R80, R90

proton_csvs = sorted(Path(".").glob("DoseZ_seed*.csv"))
if not proton_csvs:
    raise RuntimeError("No proton seed CSVs found (DoseZ_seed*.csv).")

R80_list, R90_list = [], []
for csv in proton_csvs:
    depth, dose = load_depth_dose_from_csv(csv)
    R80, R90 = compute_Rs(depth, dose)

    if np.isnan(R80) or np.isnan(R90):
        raise RuntimeError(f"{csv.name}: R80/R90 came out NaN. Check curve / binning / scoring length.")

    R80_list.append(R80)
    R90_list.append(R90)
    print(f"{csv.name}: R80={R80:.4f} cm, R90={R90:.4f} cm")

R80_mean = float(np.mean(R80_list))
R80_sd   = float(np.std(R80_list, ddof=1)) if len(R80_list) > 1 else 0.0
R90_mean = float(np.mean(R90_list))
R90_sd   = float(np.std(R90_list, ddof=1)) if len(R90_list) > 1 else 0.0

print("\nProton target (from seeds):")
print(f"R80_p = {R80_mean:.4f} ± {R80_sd:.4f} cm  (n={len(R80_list)})")
print(f"R90_p = {R90_mean:.4f} ± {R90_sd:.4f} cm  (n={len(R90_list)})")